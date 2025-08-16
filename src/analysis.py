import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from dotenv import load_dotenv
import os
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress
import requests
import time
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

def send_telegram_alert(bot_token, chat_id, message, chart_path=None):
    """Sends a message and an optional chart image to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'})
        if chart_path and os.path.exists(chart_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            files = {'photo': open(chart_path, 'rb')}
            requests.post(url, files=files, data={'chat_id': chat_id})
        logging.info(f"Telegram alert sent successfully to chat_id {chat_id}.")
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

class FyersSwingTrendlineAnalyzer:
    def __init__(self, client_id, access_token, symbol, config):
        self.client_id = client_id
        self.access_token = access_token
        self.fyers_symbol = self.format_symbol_for_fyers(symbol)
        self.display_symbol = symbol
        self.config = config
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)
        self.data = None
        self.support_lines = []
        self.resistance_lines = []
        self.last_alert_time = {}
        self.previous_price = None

    def format_symbol_for_fyers(self, symbol):
        if symbol.endswith('.NS'): return f"NSE:{symbol.replace('.NS', '')}-EQ"
        elif not (symbol.startswith('NSE:') or symbol.startswith('BSE:')): return f'NSE:{symbol}-EQ'
        return symbol

    def analyze_and_plot(self):
        """Runs the full historical analysis and generates the chart."""
        if not self.fetch_historical_data(): return False
        self.calculate_swing_trendlines()
        self.plot_chart()
        return True

    def update_and_reanalyze(self):
        """Fetches the latest data and re-runs the entire analysis."""
        logging.info(f"🔄 Re-analyzing {self.display_symbol} for new trendlines...")
        self.analyze_and_plot()

    def fetch_historical_data(self):
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=int(self.config['days_back']))
            data_request = {"symbol": self.fyers_symbol, "resolution": self.config['interval'], "date_format": "1", "range_from": from_date.strftime('%Y-%m-%d'), "range_to": to_date.strftime('%Y-%m-%d'), "cont_flag": "1"}
            logging.info(f"Fetching historical data for {self.fyers_symbol}...")
            response = self.fyers.history(data_request)
            if response['code'] == 200 and 'candles' in response and response['candles']:
                candles = response['candles']
                df_data = [{'Datetime': datetime.fromtimestamp(c[0]), 'Open': c[1], 'High': c[2], 'Low': c[3], 'Close': c[4], 'Volume': c[5]} for c in candles]
                self.data = pd.DataFrame(df_data).set_index('Datetime')
                return True
        except Exception as e:
            logging.error(f"Error fetching data for {self.fyers_symbol}: {e}")
        return False

    def _calculate_atr(self, period=14):
        if self.data is None or len(self.data) < period: return pd.Series()
        tr = pd.concat([self.data['High'] - self.data['Low'], (self.data['High'] - self.data['Close'].shift()).abs(), (self.data['Low'] - self.data['Close'].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def find_swing_points(self):
        cfg = self.config
        if self.data is None or len(self.data) < int(cfg['lookback']) + int(cfg['lookahead']) + 1: return
        highs, lows, dates = self.data['High'].values, self.data['Low'].values, self.data.index.values
        atr = self._calculate_atr().fillna(0)
        atr_ma = atr.rolling(window=50).mean().fillna(0)
        self.swing_highs, self.swing_lows = [], []
        for i in range(int(cfg['lookback']), len(highs) - int(cfg['lookahead'])):
            is_high_momentum = atr.iloc[i] > (atr_ma.iloc[i] * 0.8)
            if is_high_momentum:
                if np.all(highs[i] > highs[i-int(cfg['lookback']):i]) and np.all(highs[i] > highs[i+1:i+int(cfg['lookahead'])+1]):
                    if (highs[i] - np.mean(lows[i-int(cfg['lookback']):i+int(cfg['lookahead'])+1])) > atr.iloc[i] * float(cfg['atr_multiplier']):
                        self.swing_highs.append((i, dates[i], highs[i]))
                if np.all(lows[i] < lows[i-int(cfg['lookback']):i]) and np.all(lows[i] < lows[i+1:i+int(cfg['lookahead'])+1]):
                    if (np.mean(highs[i-int(cfg['lookback']):i+int(cfg['lookahead'])+1]) - lows[i]) > atr.iloc[i] * float(cfg['atr_multiplier']):
                        self.swing_lows.append((i, dates[i], lows[i]))
        logging.info(f"[{self.display_symbol}] Found {len(self.swing_highs)} highs and {len(self.swing_lows)} lows.")

    def _identify_primary_trend(self, num_points=5):
        if len(self.swing_lows) < 2 or len(self.swing_highs) < 2: return "RANGING"
        lows_slope, _, _, _, _ = linregress(np.arange(len(self.swing_lows[-num_points:])), [p[2] for p in self.swing_lows[-num_points:]])
        highs_slope, _, _, _, _ = linregress(np.arange(len(self.swing_highs[-num_points:])), [p[2] for p in self.swing_highs[-num_points:]])
        if lows_slope > 0 and highs_slope > 0: return "UPTREND"
        elif lows_slope < 0 and highs_slope < 0: return "DOWNTREND"
        else: return "RANGING"

    def validate_line_crossings(self, slope, intercept, max_crossings=3):
        if self.data is None: return False
        crossings = 0
        for i in range(len(self.data)):
            candle = self.data.iloc[i]
            line_price = slope * i + intercept
            body_top, body_bottom = max(candle['Open'], candle['Close']), min(candle['Open'], candle['Close'])
            if body_bottom + (candle['Close'] * 0.001) < line_price < body_top - (candle['Close'] * 0.001):
                crossings += 1
        return crossings <= max_crossings

    def find_trendlines_with_ai(self, swing_points):
        if len(swing_points) < 3: return []
        points = np.array(swing_points)
        x, y = points[:, 0].reshape(-1, 1), points[:, 2]
        try:
            ransac = RANSACRegressor(residual_threshold=(np.max(y) - np.min(y)) * 0.05, min_samples=2, max_trials=100, stop_probability=0.99)
            ransac.fit(x, y)
            inlier_mask, inliers = ransac.inlier_mask_, [p for p, is_inlier in zip(swing_points, ransac.inlier_mask_) if is_inlier]
            if len(inliers) < 3: return []
            slope, intercept = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            if self.validate_line_crossings(slope, intercept):
                score = ransac.estimator_.score(x[inlier_mask], y[inlier_mask])
                return [(slope, intercept, score, len(inliers), sorted(inliers, key=lambda p: p[0]))]
        except ValueError: pass
        return []

    def calculate_swing_trendlines(self):
        self.find_swing_points()
        primary_trend = self._identify_primary_trend()
        self.support_lines, self.resistance_lines = [], []
        if primary_trend == "UPTREND": self.support_lines = self.find_trendlines_with_ai(self.swing_lows)
        elif primary_trend == "DOWNTREND": self.resistance_lines = self.find_trendlines_with_ai(self.swing_highs)
        else:
            self.support_lines = self.find_trendlines_with_ai(self.swing_lows)
            self.resistance_lines = self.find_trendlines_with_ai(self.swing_highs)
        logging.info(f"[{self.display_symbol}] Analysis complete. Found {len(self.support_lines)} S and {len(self.resistance_lines)} R lines.")

    def plot_chart(self):
        if not self.support_lines and not self.resistance_lines: return
        style = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up='g', down='r'), gridstyle='--')
        addplots, x_range = [], range(len(self.data))
        for lines in [self.support_lines, self.resistance_lines]:
            for slope, intercept, _, _, points in lines:
                trend_y = [slope * i + intercept if i >= points[0][0] else float('nan') for i in x_range]
                addplots.append(mpf.make_addplot(trend_y, color='green' if lines == self.support_lines else 'red', width=1.5))
        os.makedirs('charts', exist_ok=True)
        chart_path = f"charts/{self.display_symbol.replace('.NS', '')}.png"
        mpf.plot(self.data, type='candle', style=style, volume=True, title=f'{self.display_symbol} Trendline Analysis', addplot=addplots, savefig=chart_path)
        logging.info(f"[{self.display_symbol}] Chart saved to {chart_path}")

    def check_for_alert(self, live_price):
        if self.data is None or self.previous_price is None:
            self.previous_price = live_price
            return
        current_index, cooldown_period = len(self.data) - 1, 3600
        for line_type, lines in [("Support", self.support_lines), ("Resistance", self.resistance_lines)]:
            for i, (slope, intercept, _, _, _) in enumerate(lines):
                line_id = f"{line_type}_{i}"
                trendline_price = slope * current_index + intercept
                if (self.previous_price > trendline_price and live_price < trendline_price) or \
                        (self.previous_price < trendline_price and live_price > trendline_price):
                    if time.time() - self.last_alert_time.get(line_id, 0) > cooldown_period:
                        self.last_alert_time[line_id] = time.time()
                        action = "crossed below" if live_price < self.previous_price else "crossed above"
                        message = f"**ALERT: {self.display_symbol} Price Crossover**\n\n`Live Price: {live_price:.2f}` just *{action}* the **{line_type} Trendline** at `~{trendline_price:.2f}`"
                        chart_path = f"charts/{self.display_symbol.replace('.NS', '')}.png"
                        send_telegram_alert(os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), message, chart_path)
                        break
        self.previous_price = live_price

# Global dictionary to hold all analyzer instances
analyzers = {}

def on_message(msg):
    """Callback function to handle incoming WebSocket messages."""
    try:
        symbol, live_price = msg.get('symbol'), msg.get('ltp')
        if symbol and live_price and symbol in analyzers:
            logging.debug(f"Live Price for {symbol}: {live_price}")
            analyzers[symbol].check_for_alert(live_price)
    except Exception as e:
        logging.error(f"Error processing message: {e}")

def main():
    config = {
        'client_id': os.getenv("CLIENT_ID"), 'access_token': os.getenv("ACCESS_TOKEN"),
        'lookback': os.getenv("LOOKBACK", "5"), 'lookahead': os.getenv("LOOKAHEAD", "5"),
        'atr_multiplier': os.getenv("ATR_MULTIPLIER", "1.5"), 'interval': os.getenv("INTERVAL", "15"),
        'days_back': os.getenv("DAYS_BACK", "60"),
        'reanalysis_interval': int(os.getenv("REANALYSIS_INTERVAL_MINUTES", "15")) * 60
    }
    if not all([config['client_id'], config['access_token'], os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")]):
        logging.error("One or more required configurations (Fyers/Telegram) are missing in .env file.")
        return

    if os.path.exists('tickers.txt'):
        with open('tickers.txt') as f: symbols = [line.strip() for line in f if line.strip()]
    else: symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS']

    # --- Phase 1: Initial Analysis ---
    logging.info("--- Starting Initial Analysis Phase ---")
    for symbol in symbols:
        analyzer = FyersSwingTrendlineAnalyzer(config['client_id'], config['access_token'], symbol, config)
        if analyzer.analyze_and_plot():
            analyzers[analyzer.fyers_symbol] = analyzer

    if not analyzers:
        logging.error("Initial analysis failed for all symbols. Exiting.")
        return

    # --- Phase 2: Real-Time Monitoring ---
    logging.info("--- Starting Real-Time Monitoring Phase ---")
    fyers_socket = data_ws.FyersDataSocket(access_token=f"{config['client_id']}:{config['access_token']}", on_message=on_message)
    fyers_socket.connect()
    fyers_socket.subscribe(symbols=list(analyzers.keys()), data_type="SymbolUpdate")

    # Run WebSocket in a background thread
    ws_thread = threading.Thread(target=fyers_socket.keep_running, daemon=True)
    ws_thread.start()

    # --- Phase 3: Periodic Re-analysis Loop ---
    while True:
        try:
            logging.info(f"Next re-analysis in {config['reanalysis_interval']} seconds...")
            time.sleep(config['reanalysis_interval'])
            for symbol, analyzer in analyzers.items():
                analyzer.update_and_reanalyze()
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Closing WebSocket.")
            fyers_socket.close_connection()
            break
        except Exception as e:
            logging.error(f"An error occurred in the re-analysis loop: {e}")

if __name__ == "__main__":
    main()