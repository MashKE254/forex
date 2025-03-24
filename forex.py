import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from functools import partial
import requests
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexSignalBot:
    def __init__(self, api_key: str, telegram_token: str, telegram_chat_id: str,
                 pairs: List[str] = None, higher_timeframe: str = "H4", 
                 lower_timeframe: str = "M30"):
        
        # Update the API initialization with environment and proper headers
        logger.info(f"setting up API-client for environment practice")
        logger.info(f"applying headers Authorization")
        self.client = API(
            access_token=api_key, 
            environment="practice",  # Explicitly set to practice or live
            headers={"Authorization": f"Bearer {api_key}"})
        
        self.telegram_bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.pairs = pairs or ['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 
                              'GBP_USD', 'GBP_JPY', 'XAU_USD', 'EUR_JPY', 
                              'AUD_JPY', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 
                              'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD']
        
        self.higher_timeframe = higher_timeframe
        self.lower_timeframe = lower_timeframe
        self.fixed_risk_amount = 100
        self.min_risk_reward = 1.75
        
        # Risk management
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.daily_loss_limit = 0.05
        self.last_check_day = datetime.now(pytz.utc).date()
        
        # Signal stats tracking
        self.rejected_signals = {}
        self.signal_attempts = {}
        
        # Initialize API connection checks and attributes
        self._check_api_connection()
        
    def _check_api_connection(self):
        """Verify API connectivity by fetching test data"""
        try:
            logger.info("Checking API connection...")
            # Test connection by making a direct request instead of using get_current_data
            
            params = {
                "count": 1,
                "granularity": "M1"
            }
            
            request = instruments.InstrumentsCandles(
                instrument="EUR_USD",
                params=params
            )
            
            logger.info(f"performing request for EUR_USD candles")
            response = self.client.request(request)
            
            # Check if the response contains candles data
            if "candles" in response and len(response["candles"]) > 0:
                logger.info("API connection successful")
            else:
                raise ConnectionError("API returned empty candles data")
                
        except Exception as e:
            logger.error(f"API connection failed: {str(e)}")
            raise RuntimeError("Failed to connect to OANDA API") from e
        
        # Candle alignment tracking
        self.last_checked = {}
        self.scheduled_checks = asyncio.Lock()
        self.active_positions = {}

    async def send_telegram_message(self, message: str) -> None:
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, 
                text=message, 
                parse_mode='HTML'
            )
            logger.info("Telegram message sent")
        except Exception as e:
            logger.error(f"Telegram error: {str(e)}")

    def get_current_data(self, pair: str, granularity: str, count: int = 500) -> pd.DataFrame:
        try:
            logger.info(f"Fetching {count} {granularity} candles for {pair}")
            params = {"count": count, "granularity": granularity}
            candles_list = []
            
            for r in InstrumentsCandlesFactory(instrument=pair, params=params):
                logger.info(f"performing request for {pair} candles")
                rv = self.client.request(r)
                candles_list.extend(rv["candles"])
            
            logger.info(f"Received {len(candles_list)} candles for {pair}")
            
            if not candles_list:
                logger.warning(f"No candles received for {pair}")
                return pd.DataFrame()
            
            prices = []
            for candle in candles_list:
                if candle['complete']:
                    prices.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c'])
                    })
            
            logger.info(f"Processed {len(prices)} complete candles for {pair}")
            
            if not prices:
                logger.warning(f"No complete candles for {pair}")
                return pd.DataFrame()
            
            df = pd.DataFrame(prices)
            if not df.empty:
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Data error {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        try:
            df = data.copy()
            
            # Trend Indicators
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

            # MACD Calculation
            fast_period = 8 if timeframe == "M30" else 12
            slow_period = 17 if timeframe == "M30" else 26
            signal_period = 6 if timeframe == "M30" else 9

            exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # ADX with +DI/-DI
            high, low, close = df['high'], df['low'], df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff().abs()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
            df['ADX'] = dx.rolling(14).mean()
            df['+DI'] = plus_di
            df['-DI'] = minus_di

            # ATR
            df['ATR'] = tr.rolling(14).mean()

            return df.ffill().dropna()

        except Exception as e:
            logger.error(f"Indicator error: {str(e)}")
            return data

    def find_swing_points(self, data: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
        """Reduced lookback to identify more recent swing points"""
        try:
            df = data.copy()
            df['SwingHigh'] = np.nan
            df['SwingLow'] = np.nan
            
            for i in range(lookback, len(df)-1):  # Changed from len(df)-lookback to len(df)-1
                # Modified to look only backward, so recent swing points can be detected
                if (df['high'].iloc[i] > df['high'].iloc[i-lookback:i]).all() and \
                   df['high'].iloc[i] > df['high'].iloc[i+1]:
                    df.iloc[i, df.columns.get_loc('SwingHigh')] = df['high'].iloc[i]
                
                if (df['low'].iloc[i] < df['low'].iloc[i-lookback:i]).all() and \
                   df['low'].iloc[i] < df['low'].iloc[i+1]:
                    df.iloc[i, df.columns.get_loc('SwingLow')] = df['low'].iloc[i]
            
            return df
        
        except Exception as e:
            logger.error(f"Swing point error: {str(e)}")
            return data

    def check_daily_loss_limit(self) -> bool:
        current_day = datetime.now(pytz.utc).date()
        if current_day != self.last_check_day:
            self.last_check_day = current_day
            self.daily_start_balance = self.current_balance
        
        daily_pnl = self.current_balance - self.daily_start_balance
        if daily_pnl <= -self.daily_start_balance * self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached ({daily_pnl:.2f})")
            return True
        return False

    def format_signal_message(self, pair: str, signal_type: str, entry_price: float, 
                            stop_loss: float, take_profit: float) -> str:
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        return f"""
🔔 <b>New {pair} Signal</b>

Type: {'🟢 BUY' if signal_type == 'LONG' else '🔴 SELL'}
Entry Price: {entry_price:.5f}
Stop Loss: {stop_loss:.5f}
Take Profit: {take_profit:.5f}
Risk/Reward: {risk_reward:.2f}
Position Size: {position_size:.2f} units

⚠️ <i>Fixed risk: ${self.fixed_risk_amount} per trade</i>
"""

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_per_unit = abs(entry_price - stop_loss)
        return self.fixed_risk_amount / risk_per_unit if risk_per_unit != 0 else 0

    def log_rejected_signal(self, pair: str, signal_type: str, reason: str, data=None):
        """Track and log rejected signals to help diagnose issues"""
        if pair not in self.rejected_signals:
            self.rejected_signals[pair] = []
        
        rejection = {
            'time': datetime.now(pytz.utc),
            'type': signal_type,
            'reason': reason
        }
        
        if data is not None:
            rejection['data'] = {
                'EMA50': data.get('EMA50'),
                'EMA200': data.get('EMA200'),
                'h_tf_EMA50': data.get('h_tf_EMA50'),
                'h_tf_EMA200': data.get('h_tf_EMA200'),
                'MACD': data.get('MACD'),
                'MACD_Hist': data.get('MACD_Hist'),
                'prev_MACD_Hist': data.get('prev_MACD_Hist'),
                'RSI': data.get('RSI'),
                'ADX': data.get('ADX')
            }
        
        self.rejected_signals[pair].append(rejection)
        logger.info(f"Signal rejected for {pair} ({signal_type}): {reason}")

    async def check_for_signals(self, pair: str) -> None:
        try:     
            # Track how many times we attempt to check signals
            if pair not in self.signal_attempts:
                self.signal_attempts[pair] = 0
            self.signal_attempts[pair] += 1
            
            # Rate limiting - reduced from 5 minutes to 2 minutes
            async with self.scheduled_checks:
                now = datetime.now(pytz.utc)
                if pair in self.last_checked:
                    elapsed = now - self.last_checked[pair]
                    if elapsed < timedelta(minutes=2):  # Reduced from 5 to 2 minutes
                        return
                
                # Always update the last checked time
                self.last_checked[pair] = now

                if self.check_daily_loss_limit():
                    return

                data_lower = self.get_current_data(pair, self.lower_timeframe)
                data_higher = self.get_current_data(pair, self.higher_timeframe)
                
                if data_lower.empty or data_higher.empty:
                    logger.warning(f"No data received for {pair}")
                    return

                data_lower = self.calculate_indicators(data_lower, self.lower_timeframe)
                data_higher = self.calculate_indicators(data_higher, self.higher_timeframe)
                
                # Add validation
                if len(data_higher) < 200 or data_higher['EMA200'].isna().any():
                    logger.error(f"Insufficient H4 data for {pair} - only {len(data_higher)} candles")
                    return
                
                h_tf_ema50 = f"{self.higher_timeframe}_EMA50"
                h_tf_ema200 = f"{self.higher_timeframe}_EMA200"
                
                for idx in data_lower.index:
                    matching_higher_idx = data_higher.index.asof(idx)
                    if not pd.isna(matching_higher_idx):
                        data_lower.at[idx, h_tf_ema50] = data_higher.at[matching_higher_idx, 'EMA50']
                        data_lower.at[idx, h_tf_ema200] = data_higher.at[matching_higher_idx, 'EMA200']
                
                data = data_lower.ffill().dropna()
                
                # Check if we have enough rows after applying indicators
                if len(data) < 50:
                    logger.warning(f"Insufficient processed data for {pair}: only {len(data)} rows")
                    return
                
                # Find swing points with reduced lookback period
                data = self.find_swing_points(data, lookback=10)
                
                current_bar = data.iloc[-1]
                prev_bar = data.iloc[-2]
                
                # Debugging logging - more detailed
                debug_info = f"""
                {pair} Signal Check (attempt #{self.signal_attempts[pair]}):
                EMA50 > EMA200: {current_bar['EMA50'] > current_bar['EMA200']} (values: {current_bar['EMA50']:.5f} vs {current_bar['EMA200']:.5f})
                HTF EMA50 > HTF EMA200: {current_bar[h_tf_ema50] > current_bar[h_tf_ema200]} (values: {current_bar[h_tf_ema50]:.5f} vs {current_bar[h_tf_ema200]:.5f})
                MACD: {current_bar['MACD']:.6f}, Signal: {current_bar['Signal']:.6f}
                MACD Hist: {current_bar['MACD_Hist']:.6f}, Previous: {prev_bar['MACD_Hist']:.6f}
                MACD Hist > 0: {current_bar['MACD_Hist'] > 0}
                MACD Hist crossover: {current_bar['MACD_Hist'] > 0 and prev_bar['MACD_Hist'] <= 0}
                ADX: {current_bar['ADX']:.2f} (needs to be > 25)
                RSI: {current_bar['RSI']:.2f}
                Recent SwingLows: {len(data['SwingLow'].iloc[-20:].dropna())} points
                Recent SwingHighs: {len(data['SwingHigh'].iloc[-20:].dropna())} points
                """
                logger.info(debug_info)

                # Track all conditions for debugging
                long_conditions = {
                    'ema_alignment': current_bar['EMA50'] > current_bar['EMA200'],
                    'h_tf_ema_alignment': current_bar[h_tf_ema50] > current_bar[h_tf_ema200],
                    'macd_hist_current': current_bar['MACD_Hist'] > 0,
                    'macd_hist_prev': prev_bar['MACD_Hist'] <= 0,
                    'macd_value': current_bar['MACD'] < 0,
                    'rsi': current_bar['RSI'] < 65,
                    'adx': current_bar['ADX'] > 25
                }
                
                short_conditions = {
                    'ema_alignment': current_bar['EMA50'] < current_bar['EMA200'],
                    'h_tf_ema_alignment': current_bar[h_tf_ema50] < current_bar[h_tf_ema200],
                    'macd_hist_current': current_bar['MACD_Hist'] < 0,
                    'macd_hist_prev': prev_bar['MACD_Hist'] >= 0,
                    'macd_value': current_bar['MACD'] > 0,
                    'rsi': current_bar['RSI'] > 35,
                    'adx': current_bar['ADX'] > 25
                }

                # Log condition results for each pair
                logger.info(f"LONG conditions for {pair}: {long_conditions}")
                logger.info(f"SHORT conditions for {pair}: {short_conditions}")

                # Long Signal
                if all(long_conditions.values()):
                    recent_lows = data['SwingLow'].iloc[-20:].dropna()
                    
                    if recent_lows.empty:
                        self.log_rejected_signal(pair, "LONG", "No recent swing lows found")
                    else:
                        stop_loss = recent_lows.iloc[-1]
                        entry_price = current_bar['close']
                        risk = entry_price - stop_loss
                        
                        if risk <= 0:
                            self.log_rejected_signal(pair, "LONG", f"Invalid risk: {risk} (entry: {entry_price}, SL: {stop_loss})")
                        else:
                            take_profit = entry_price + (risk * self.min_risk_reward)
                            
                            logger.info(f"LONG SIGNAL FOUND for {pair} - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                            message = self.format_signal_message(pair, "LONG", entry_price, stop_loss, take_profit)
                            await self.send_telegram_message(message)
                            
                            # Track position
                            self.active_positions[pair] = {
                                'type': 'LONG',
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'size': self.calculate_position_size(entry_price, stop_loss)
                            }
                else:
                    # Find which conditions failed
                    failed_conditions = [k for k, v in long_conditions.items() if not v]
                    self.log_rejected_signal(
                        pair, 
                        "LONG", 
                        f"Failed conditions: {', '.join(failed_conditions)}",
                        {
                            'EMA50': current_bar['EMA50'],
                            'EMA200': current_bar['EMA200'],
                            'h_tf_EMA50': current_bar[h_tf_ema50],
                            'h_tf_EMA200': current_bar[h_tf_ema200],
                            'MACD': current_bar['MACD'],
                            'MACD_Hist': current_bar['MACD_Hist'],
                            'prev_MACD_Hist': prev_bar['MACD_Hist'],
                            'RSI': current_bar['RSI'],
                            'ADX': current_bar['ADX']
                        }
                    )

                # Short Signal
                if all(short_conditions.values()):
                    recent_highs = data['SwingHigh'].iloc[-20:].dropna()
                    
                    if recent_highs.empty:
                        self.log_rejected_signal(pair, "SHORT", "No recent swing highs found")
                    else:
                        stop_loss = recent_highs.iloc[-1]
                        entry_price = current_bar['close']
                        risk = stop_loss - entry_price
                        
                        if risk <= 0:
                            self.log_rejected_signal(pair, "SHORT", f"Invalid risk: {risk} (entry: {entry_price}, SL: {stop_loss})")
                        else:
                            take_profit = entry_price - (risk * self.min_risk_reward)
                            
                            logger.info(f"SHORT SIGNAL FOUND for {pair} - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                            message = self.format_signal_message(pair, "SHORT", entry_price, stop_loss, take_profit)
                            await self.send_telegram_message(message)
                            
                            self.active_positions[pair] = {
                                'type': 'SHORT',
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'size': self.calculate_position_size(entry_price, stop_loss)
                            }
                else:
                    # Find which conditions failed
                    failed_conditions = [k for k, v in short_conditions.items() if not v]
                    self.log_rejected_signal(
                        pair, 
                        "SHORT", 
                        f"Failed conditions: {', '.join(failed_conditions)}",
                        {
                            'EMA50': current_bar['EMA50'],
                            'EMA200': current_bar['EMA200'],
                            'h_tf_EMA50': current_bar[h_tf_ema50],
                            'h_tf_EMA200': current_bar[h_tf_ema200],
                            'MACD': current_bar['MACD'],
                            'MACD_Hist': current_bar['MACD_Hist'],
                            'prev_MACD_Hist': prev_bar['MACD_Hist'],
                            'RSI': current_bar['RSI'],
                            'ADX': current_bar['ADX']
                        }
                    )

                # Check existing positions
                if pair in self.active_positions:
                    position = self.active_positions[pair]
                    current_low = current_bar['low']
                    current_high = current_bar['high']
                    
                    if position['type'] == 'LONG':
                        if current_low <= position['sl']:
                            pl = (position['sl'] - position['entry']) * position['size']
                            self.current_balance += pl
                            logger.info(f"Position stopped out: {pair} LONG at {position['sl']}, P&L: {pl:.2f}")
                            await self.send_telegram_message(f"⚠️ Stop Loss hit on {pair} LONG. P&L: ${pl:.2f}")
                            del self.active_positions[pair]
                        elif current_high >= position['tp']:
                            pl = (position['tp'] - position['entry']) * position['size']
                            self.current_balance += pl
                            logger.info(f"Take profit hit: {pair} LONG at {position['tp']}, P&L: {pl:.2f}")
                            await self.send_telegram_message(f"✅ Take Profit hit on {pair} LONG. P&L: ${pl:.2f}")
                            del self.active_positions[pair]
                    
                    else:  # SHORT
                        if current_high >= position['sl']:
                            pl = (position['entry'] - position['sl']) * position['size']
                            self.current_balance += pl
                            logger.info(f"Position stopped out: {pair} SHORT at {position['sl']}, P&L: {pl:.2f}")
                            await self.send_telegram_message(f"⚠️ Stop Loss hit on {pair} SHORT. P&L: ${pl:.2f}")
                            del self.active_positions[pair]
                        elif current_low <= position['tp']:
                            pl = (position['entry'] - position['tp']) * position['size']
                            self.current_balance += pl
                            logger.info(f"Take profit hit: {pair} SHORT at {position['tp']}, P&L: {pl:.2f}")
                            await self.send_telegram_message(f"✅ Take Profit hit on {pair} SHORT. P&L: ${pl:.2f}")
                            del self.active_positions[pair]

        except Exception as e:
            logger.error(f"Signal check error {pair}: {str(e)}")
            
    async def send_signal_stats(self):
        """Send signal statistics to Telegram"""
        stats_message = """
<b>🔍 Signal Scanner Statistics</b>

<u>Signal Attempts:</u>
"""
        for pair, count in self.signal_attempts.items():
            stats_message += f"{pair}: {count} checks\n"
        
        stats_message += "\n<u>Recent Rejected Signals:</u>\n"
        
        recent_rejections = []
        for pair, rejections in self.rejected_signals.items():
            for rejection in rejections[-3:]:  # Get last 3 rejections per pair
                recent_rejections.append(f"{pair} {rejection['type']}: {rejection['reason']}")
        
        if recent_rejections:
            stats_message += "\n".join(recent_rejections[-10:])  # Show last 10 rejections overall
        else:
            stats_message += "No rejected signals yet."
            
        stats_message += f"\n\n<i>Current balance: ${self.current_balance:.2f}</i>"
        
        await self.send_telegram_message(stats_message)
            
app = FastAPI()
forex_bot = ForexSignalBot(
    api_key=os.getenv("OANDA_API_KEY"),
    telegram_token=os.getenv("TELEGRAM_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID")
)

# Create a single global scheduler
scheduler = AsyncIOScheduler(timezone="UTC")

# Create the async functions that will be directly called by the scheduler
async def check_all_signals_job():
    """Check all currency pairs"""
    tasks = [forex_bot.check_for_signals(pair) for pair in forex_bot.pairs]
    await asyncio.gather(*tasks)
    logger.info("Completed full signal check at %s", datetime.utcnow())

async def check_h4_confirmation_job():
    """Update higher timeframe trends"""
    logger.info("Updating H4 trend confirmations")
    # Add H4-specific logic here

async def send_daily_stats_job():
    """Send daily statistics report"""
    await forex_bot.send_signal_stats()
    logger.info("Sent daily stats report")

@app.on_event("startup")
async def startup_event():
    # Configure scheduler with explicit job store and executor
    job_defaults = {
        'misfire_grace_time': 300,
        'coalesce': True,
        'max_instances': 1
    }
    
    scheduler = AsyncIOScheduler(
        job_defaults=job_defaults,
        timezone="UTC"
    )

    # Check more frequently - every 15 minutes instead of 30
    scheduler.add_job(
        check_all_signals_job,
        CronTrigger(minute="0,15,30,45", second="5"),
        name="15m_signals_check"
    )
    
    scheduler.add_job(
        check_h4_confirmation_job,
        CronTrigger(hour="0,4,8,12,16,20", minute="0", second="10"),
        name="4h_trend_update"
    )
    
    # Add a daily stats report job
    scheduler.add_job(
        send_daily_stats_job,
        CronTrigger(hour="20", minute="0", second="0"),
        name="daily_stats_report"
    )
    
    # Start the scheduler
    scheduler.start()
    
    # Send startup notification
    await forex_bot.send_telegram_message("🚀 Forex Signal Bot started! Monitoring markets...")
    logger.info("Forex Signal Bot started")

@app.on_event("shutdown")
async def shutdown_event():
    # Shutdown notification
    try:
        await forex_bot.send_telegram_message("⚠️ Forex Signal Bot shutting down...")
    except:
        pass
    
    # Shut down the scheduler
    scheduler.shutdown()
    logger.info("Forex Signal Bot shut down")
    
@app.get("/")
async def root():
    return {"message": "Forex Signal Bot API is running."}


@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the service is running"""
    return {"status": "ok", "time": datetime.now(pytz.utc).isoformat()}

@app.get("/stats")
async def get_stats():
    """Return statistics about the signal bot"""
    return {
        "active_positions": forex_bot.active_positions,
        "signal_attempts": forex_bot.signal_attempts,
        "rejected_signals_count": {pair: len(rejections) for pair, rejections in forex_bot.rejected_signals.items()},
        "current_balance": forex_bot.current_balance,
        "daily_start_balance": forex_bot.daily_start_balance,
        "profit_loss": forex_bot.current_balance - forex_bot.initial_balance
    }

@app.post("/trigger-scan")
async def trigger_scan():
    """Manually trigger a signal scan across all pairs"""
    await check_all_signals_job()
    return {"status": "scan completed", "time": datetime.now(pytz.utc).isoformat()}

@app.post("/send-stats")
async def send_stats():
    """Manually trigger sending of statistics"""
    await send_daily_stats_job()
    return {"status": "stats sent", "time": datetime.now(pytz.utc).isoformat()}

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        "forex:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )