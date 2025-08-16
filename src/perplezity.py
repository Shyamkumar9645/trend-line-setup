import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from dotenv import load_dotenv
import os
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress

# Load environment variables from .env file
load_dotenv()

class FyersSwingTrendlineAnalyzer:
    def __init__(self, client_id, access_token, symbol, interval='1'):
        """
        Initialize the FyersSwingTrendlineAnalyzer.
        """
        self.client_id = client_id
        self.access_token = access_token
        self.symbol = symbol
        self.interval = interval
        self.fyers = fyersModel.FyersModel(client_id=client_id, token=access_token)

        # Data storage
        self.data = None
        self.swing_highs = []
        self.swing_lows = []
        self.support_lines = []
        self.resistance_lines = []

    def format_symbol_for_fyers(self, symbol):
        """
        Convert a common stock symbol to the Fyers API format.
        """
        if symbol.endswith('.NS'):
            return f"NSE:{symbol.replace('.NS', '')}-EQ"
        elif symbol.startswith('NSE:') or symbol.startswith('BSE:'):
            return symbol
        else:
            return f'NSE:{symbol}-EQ'

    def fetch_historical_data(self, days_back=60):
        """
        Fetch historical data using the Fyers API.
        """
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            from_date_str = from_date.strftime('%Y-%m-%d')
            to_date_str = to_date.strftime('%Y-%m-%d')
            data_request = {"symbol": self.symbol, "resolution": self.interval, "date_format": "1", "range_from": from_date_str, "range_to": to_date_str, "cont_flag": "1"}
            print(f"Fetching historical data for {self.symbol}...")
            response = self.fyers.history(data_request)
            if response['code'] == 200 and 'candles' in response and response['candles']:
                candles = response['candles']
                df_data = [{'Datetime': datetime.fromtimestamp(c[0]), 'Open': c[1], 'High': c[2], 'Low': c[3], 'Close': c[4], 'Volume': c[5]} for c in candles]
                self.data = pd.DataFrame(df_data).set_index('Datetime')
                print(f"Successfully fetched {len(self.data)} candles.")
                return True
            else:
                print(f"Error fetching data: {response.get('message', 'No data received')}")
                return False
        except Exception as e:
            print(f"An error occurred during data fetching: {e}")
            return False

    def _calculate_atr(self, period=14):
        """
        Calculate the Average True Range (ATR).
        """
        if self.data is None or len(self.data) < period: return None
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def find_swing_points(self, lookback=3, lookahead=3, atr_multiplier=1.5):
        """
        Identifies significant swing points, with a volatility filter to ignore
        swings that form in low-momentum, consolidated areas.
        """
        if self.data is None or len(self.data) < lookback + lookahead + 1:
            return

        highs = self.data['High'].values
        lows = self.data['Low'].values
        dates = self.data.index.values
        atr = self._calculate_atr().fillna(0)

        # Calculate a baseline for "normal" volatility using a 50-period moving average of ATR
        atr_ma = atr.rolling(window=50).mean().fillna(0)

        self.swing_highs = []
        self.swing_lows = []

        for i in range(lookback, len(highs) - lookahead):
            # VOLATILITY CHECK: Is the volatility at this point high enough?
            # We consider it high enough if it's at least 80% of the recent average.
            is_high_momentum = atr.iloc[i] > (atr_ma.iloc[i] * 0.8)

            # Check for swing high
            is_swing_high = np.all(highs[i] > highs[i-lookback:i]) and np.all(highs[i] > highs[i+1:i+lookahead+1])
            if is_swing_high and is_high_momentum:
                if (highs[i] - np.mean(lows[i-lookback:i+lookahead+1])) > atr.iloc[i] * atr_multiplier:
                    self.swing_highs.append((i, dates[i], highs[i]))

            # Check for swing low
            is_swing_low = np.all(lows[i] < lows[i-lookback:i]) and np.all(lows[i] < lows[i+1:i+lookahead+1])
            if is_swing_low and is_high_momentum:
                if (np.mean(highs[i-lookback:i+lookahead+1]) - lows[i]) > atr.iloc[i] * atr_multiplier:
                    self.swing_lows.append((i, dates[i], lows[i]))

        print(f"Found {len(self.swing_highs)} high-momentum swing highs and {len(self.swing_lows)} high-momentum swing lows.")

    def _identify_primary_trend(self, num_points=5):
        """
        Analyzes the sequence of the last few swing points to determine the primary trend.
        """
        if len(self.swing_lows) < 2 or len(self.swing_highs) < 2:
            return "RANGING"
        recent_lows = np.array([p[2] for p in self.swing_lows[-num_points:]])
        lows_x = np.arange(len(recent_lows))
        lows_slope, _, _, _, _ = linregress(lows_x, recent_lows)
        recent_highs = np.array([p[2] for p in self.swing_highs[-num_points:]])
        highs_x = np.arange(len(recent_highs))
        highs_slope, _, _, _, _ = linregress(highs_x, recent_highs)

        if lows_slope > 0 and highs_slope > 0:
            return "UPTREND"
        elif lows_slope < 0 and highs_slope < 0:
            return "DOWNTREND"
        else:
            return "RANGING"

    def validate_line_crossings(self, slope, intercept, max_crossings=3):
        """
        Validates a trendline using a dynamic, price-based tolerance.
        """
        if self.data is None: return False
        significant_crossings = 0
        for i in range(len(self.data)):
            candle = self.data.iloc[i]
            line_price_at_candle = slope * i + intercept
            body_top = max(candle['Open'], candle['Close'])
            body_bottom = min(candle['Open'], candle['Close'])
            tolerance = candle['Close'] * 0.001
            if (line_price_at_candle > body_bottom + tolerance and
                    line_price_at_candle < body_top - tolerance):
                significant_crossings += 1
        return significant_crossings <= max_crossings

    def find_trendlines_with_ai(self, swing_points):
        """
        Finds the single best trendline for a set of points, then validates it.
        """
        if len(swing_points) < 3: return []
        points = np.array(swing_points)
        x, y = points[:, 0].reshape(-1, 1), points[:, 2]
        try:
            price_range = np.max(y) - np.min(y)
            if price_range == 0: return []
            threshold = price_range * 0.05
            ransac = RANSACRegressor(residual_threshold=threshold, min_samples=2, max_trials=100, stop_probability=0.99)
            ransac.fit(x, y)
            inlier_mask = ransac.inlier_mask_
            inliers = [p for p, is_inlier in zip(swing_points, inlier_mask) if is_inlier]
            if len(inliers) < 3: return []
            slope, intercept = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
            if self.validate_line_crossings(slope, intercept):
                score = ransac.estimator_.score(x[inlier_mask], y[inlier_mask])
                return [(slope, intercept, score, len(inliers), sorted(inliers, key=lambda p: p[0]))]
            else:
                print("🚫 AI-proposed line was rejected due to excessive price crossings.")
                return []
        except ValueError:
            return []

    def calculate_swing_trendlines(self, lookback, lookahead, atr_multiplier):
        """
        Orchestrates an intelligent analysis: first identify the trend, then draw the
        single most relevant trendline for that trend.
        """
        if self.data is None: return
        self.find_swing_points(lookback, lookahead, atr_multiplier)
        primary_trend = self._identify_primary_trend()
        self.support_lines, self.resistance_lines = [], []

        if primary_trend == "UPTREND":
            print("📈 Uptrend detected. Focusing on the support trendline.")
            self.support_lines = self.find_trendlines_with_ai(self.swing_lows)
        elif primary_trend == "DOWNTREND":
            print("📉 Downtrend detected. Focusing on the resistance trendline.")
            self.resistance_lines = self.find_trendlines_with_ai(self.swing_highs)
        else:
            print("📊 Ranging market detected. Analyzing both potential boundaries.")
            self.support_lines = self.find_trendlines_with_ai(self.swing_lows)
            self.resistance_lines = self.find_trendlines_with_ai(self.swing_highs)

        print(f"Found {len(self.support_lines)} valid support and {len(self.resistance_lines)} valid resistance lines.")

    def print_analysis_summary(self):
        if self.data is None: return
        print(f"\n{'='*60}\nAI TRENDLINE ANALYSIS SUMMARY\n{'='*60}")
        current_price = self.data['Close'].iloc[-1]
        current_index = len(self.data) - 1
        print(f"Symbol: {self.symbol}\nCurrent Price: ₹{current_price:.2f}")

        if self.support_lines:
            slope, intercept, r2, touches, _ = self.support_lines[0]
            current_level = slope * current_index + intercept
            distance_pct = (current_price - current_level) / current_price * 100
            print(f"\n📈 SUPPORT LINE:\n   Current Level: ₹{current_level:.2f}")
            print(f"   Proximity: Price is {distance_pct:.2f}% {'above' if distance_pct > 0 else 'below'} support.")
            print(f"   Strength: Based on {touches} swing points with R² of {r2:.3f}")
        else:
            print(f"\n📈 SUPPORT LINE: No valid trendline found.")

        if self.resistance_lines:
            slope, intercept, r2, touches, _ = self.resistance_lines[0]
            current_level = slope * current_index + intercept
            distance_pct = (current_level - current_price) / current_price * 100
            print(f"\n📉 RESISTANCE LINE:\n   Current Level: ₹{current_level:.2f}")
            print(f"   Proximity: Price is {distance_pct:.2f}% below resistance.")
            print(f"   Strength: Based on {touches} swing points with R² of {r2:.3f}")
        else:
            print(f"\n📉 RESISTANCE LINE: No valid trendline found.")

    def plot_chart(self, figsize=(16, 8)):
        if not self.support_lines and not self.resistance_lines:
            print("\nNo valid trendlines to display. Skipping chart.")
            return
        if self.data is None: return
        style = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up='g', down='r'), gridstyle='--')
        addplots = []
        x_range = range(len(self.data))

        for slope, intercept, _, _, points in self.support_lines:
            start_idx = points[0][0]
            trend_y = [slope * i + intercept if i >= start_idx else float('nan') for i in x_range]
            addplots.append(mpf.make_addplot(trend_y, color='green', width=1.5))

        for slope, intercept, _, _, points in self.resistance_lines:
            start_idx = points[0][0]
            trend_y = [slope * i + intercept if i >= start_idx else float('nan') for i in x_range]
            addplots.append(mpf.make_addplot(trend_y, color='red', width=1.5))

        print("\nDisplaying chart... Close the chart window to continue.")
        mpf.plot(self.data, type='candle', style=style, volume=True,
                 title=f'{self.symbol} - Intelligent AI Trendline Analysis',
                 addplot=addplots if addplots else None, figsize=figsize)

def get_stock_config():
    return {"lookback": 5, "lookahead": 5, "atr_multiplier": 1.5}

def quick_fyers_analysis(client_id, access_token, symbol, interval='60', days_back=90):
    print("-" * 80)
    print(f"Starting AI Analysis for: {symbol}")
    analyzer = FyersSwingTrendlineAnalyzer(client_id, access_token, symbol, interval)
    analyzer.symbol = analyzer.format_symbol_for_fyers(symbol)
    if analyzer.fetch_historical_data(days_back):
        config = get_stock_config()
        analyzer.calculate_swing_trendlines(
            lookback=config["lookback"],
            lookahead=config["lookahead"],
            atr_multiplier=config["atr_multiplier"]
        )
        analyzer.print_analysis_summary()
        analyzer.plot_chart()
    else:
        print(f"Failed to perform analysis for {symbol} due to data fetching issues.")

def main():
    CLIENT_ID = os.getenv("CLIENT_ID")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    if not CLIENT_ID or not ACCESS_TOKEN:
        print("\n❌ ERROR: CLIENT_ID or ACCESS_TOKEN not found in .env file.")
        return

    if os.path.exists('tickers.txt'):
        with open('tickers.txt') as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"\nLoaded {len(symbols)} symbols from tickers.txt")
    else:
        symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS']
        print(f"\n'tickers.txt' not found. Using default symbols: {', '.join(symbols)}")

    for symbol in symbols:
        try:
            # You can configure the timeframe and historical data period here
            quick_fyers_analysis(CLIENT_ID, ACCESS_TOKEN, symbol, interval='60', days_back=99)
        except Exception as e:
            print(f"An unexpected error occurred while analyzing {symbol}: {e}")

if __name__ == "__main__":
    print("🚀 Fyers Intelligent High-Momentum Trend Analyzer 🚀")
    print("=" * 80)
    print("This script identifies the primary trend and draws the most relevant, high-momentum trendline.")
    print("Required packages: pip install pandas numpy matplotlib mplfinance fyers-apiv3 python-dotenv scikit-learn scipy")
    print("=" * 80)
    main()