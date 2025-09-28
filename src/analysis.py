import pandas as pd
import mplfinance as mpf
from fyers_apiv3 import fyersModel
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import os
import sys
import time
from dotenv import load_dotenv
import requests
import calendar

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.websocket_client import FyersWebsocketClient

# --- Load Configuration from .env file ---
load_dotenv(override=True)

# --- Fyers API Configuration ---
CLIENT_ID = os.getenv("CLIENT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# --- Telegram Alerting Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

# --- Analysis & Alerting Parameters ---
INTERVAL = os.getenv("INTERVAL", "30")
DAYS_BACK = int(os.getenv("DAYS_BACK", 25))
ALERT_THRESHOLD_PERCENT = float(os.getenv("ALERT_THRESHOLD_PERCENT", 1.5))

print(f"✅ Script is using INTERVAL: {INTERVAL}")
print(f"✅ Script is using DAYS_BACK: {DAYS_BACK}")

# --- Main Symbol List ---
base_indexes = [
    'NSE:NIFTY50-INDEX',
    'NSE:NIFTYBANK-INDEX',
    'BSE:SENSEX-INDEX'
]

env_tickers = []
tickers_str = os.getenv("TICKERS")
if tickers_str:
    env_tickers = [ticker.strip() for ticker in tickers_str.split(',')]
    print(f"✅ Loaded {len(env_tickers)} additional tickers from .env file.")
else:
    print("ℹ️ No additional tickers found in .env file.")

tickers = list(dict.fromkeys(base_indexes + env_tickers))
print(f"➡️ Total unique symbols to be analyzed: {len(tickers)}")


# --- Fyers Model Initialization ---
try:
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, log_path=os.getcwd())
    print("✅ FyersModel initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing FyersModel: {e}")
    exit()

# --- Helper Functions ---
def send_telegram_alert(message, image_path=None):
    if not TELEGRAM_ENABLED: return
    try:
        if image_path:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
                response = requests.post(url, data=data, files=files)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, json=payload)
        if response.status_code == 200: print("✅ Telegram alert sent successfully.")
        else: print(f"❌ Failed to send Telegram alert: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ An error occurred while sending Telegram alert: {e}")

def get_atm_strike(spot_price, strike_interval=50):
    return int(round(spot_price / strike_interval) * strike_interval)

HOLIDAYS_2025 = [
    datetime(2025, 1, 1).date(),
    datetime(2025, 2, 26).date(),
    datetime(2025, 3, 14).date(),
    datetime(2025, 3, 31).date(),
    datetime(2025, 4, 10).date(),
    datetime(2025, 4, 14).date(),
    datetime(2025, 4, 18).date(),
    datetime(2025, 5, 1).date(),
    datetime(2025, 8, 15).date(),
    datetime(2025, 8, 27).date(),
    datetime(2025, 10, 2).date(),
    datetime(2025, 10, 21).date(),
    datetime(2025, 10, 22).date(),
    datetime(2025, 11, 5).date(),
    datetime(2025, 12, 25).date(),
]

def get_next_weekly_expiry(expiry_weekday):
    today = datetime.now()
    days_ahead = expiry_weekday - today.weekday()
    if days_ahead < 0 or (days_ahead == 0 and today.hour >= 16):
        days_ahead += 7
    
    next_expiry = today + timedelta(days=days_ahead)
    
    # Adjust for holidays
    while next_expiry.date() in HOLIDAYS_2025 or next_expiry.weekday() >= 5: # 5=Saturday, 6=Sunday
        next_expiry -= timedelta(days=1)
        
    return next_expiry

def get_monthly_expiry(year, month):
    _, last_day = calendar.monthrange(year, month)
    last_date = datetime(year, month, last_day)
    offset = (last_date.weekday() - 3) % 7 # 3 is for Thursday
    monthly_expiry = last_date - timedelta(days=offset)
    
    # Adjust for holidays
    while monthly_expiry.date() in HOLIDAYS_2025 or monthly_expiry.weekday() >= 5:
        monthly_expiry -= timedelta(days=1)
        
    return monthly_expiry

def get_fyers_option_symbol(index_name, expiry_date, strike, option_type, expiry_type):
    expiry_yy = expiry_date.strftime('%y')

    if expiry_type == "WEEKLY":
        month_code = str(expiry_date.month) if expiry_date.month < 10 else {10: 'O', 11: 'N', 12: 'D'}[expiry_date.month]
        expiry_mon = month_code
        expiry_dd = expiry_date.strftime('%d')
    else: # MONTHLY
        expiry_mon = expiry_date.strftime('%b').upper()
        expiry_dd = ""

    symbol_map = { 'NIFTY 50': 'NIFTY', 'NIFTY BANK': 'BANKNIFTY', 'NIFTY FIN SERVICE': 'FINNIFTY', 'SENSEX': 'SENSEX' }
    base_symbol = symbol_map.get(index_name)
    exchange = "BSE" if index_name == "SENSEX" else "NSE"
    if not base_symbol: return None
    return f"{exchange}:{base_symbol}{expiry_yy}{expiry_mon}{expiry_dd}{strike}{option_type.upper()}"

def calculate_atr(df, period=14):
    """Calculates the Average True Range (ATR) for a given DataFrame."""
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df


# --- Trendline Detection Parameters ---
MIN_TOUCHES = int(os.getenv("MIN_TOUCHES", "3"))
MAX_DEVIATION_PERCENT = float(os.getenv("MAX_DEVIATION_PERCENT", "0.01")) # 1% deviation allowed for touches
MAX_CROSSINGS = int(os.getenv("MAX_CROSSINGS", "2")) # Max allowed close price crosses
PROMINENCE_PERCENTILE = int(os.getenv("PROMINENCE_PERCENTILE", "75")) # Only consider peaks above this prominence percentile
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "1.0")) # Multiplier for ATR to calculate price tolerance

print(f"✅ Trendline MIN_TOUCHES: {MIN_TOUCHES}")
print(f"✅ Trendline MAX_DEVIATION_PERCENT: {MAX_DEVIATION_PERCENT}")
print(f"✅ Trendline MAX_CROSSINGS: {MAX_CROSSINGS}")
print(f"✅ Trendline PROMINENCE_PERCENTILE: {PROMINENCE_PERCENTILE}")
print(f"✅ Trendline ATR_MULTIPLIER: {ATR_MULTIPLIER}")

def _get_line_equation(p1_x, p1_y, p2_x, p2_y):
    """Calculates the slope and y-intercept of a line given two points."""
    if p1_x == p2_x: # Vertical line, should ideally not happen with time-series data
        return None, None
    slope = (p2_y - p1_y) / (p2_x - p1_x)
    intercept = p1_y - slope * p1_x
    return slope, intercept

def _get_y_on_line(slope, intercept, x):
    """Calculates the y-value on a line for a given x."""
    if slope is None or intercept is None: return None
    return slope * x + intercept

def find_trendlines(df, is_support=True, atr_multiplier=1.0):
    price_data = df['Low'].values if is_support else df['High'].values
    # Invert for resistance to find peaks in 'High' values
    if not is_support:
        price_data = df['High'].values

    # Find peaks (swing points)
    # For support, we look for valleys, so we invert the price data
    # For resistance, we look for peaks in the high values
    peaks_indices, properties = find_peaks(price_data if not is_support else -price_data, prominence=1)

    if len(peaks_indices) < MIN_TOUCHES:
        return None, None

    # Filter peaks by prominence percentile
    prominences = properties['prominences']
    if len(prominences) > 0:
        prominence_threshold = np.percentile(prominences, PROMINENCE_PERCENTILE)
        significant_peaks_indices = [idx for idx, prom in zip(peaks_indices, prominences) if prom >= prominence_threshold]
    else:
        significant_peaks_indices = peaks_indices

    if len(significant_peaks_indices) < MIN_TOUCHES:
        return None, None

    significant_points_df = df.iloc[significant_peaks_indices]
    x_axis = np.arange(len(df.index))

    best_line_coeffs = None
    best_score = -1
    best_line_points = None
    best_touches = -1
    best_r_squared = -1

    # Iterate through all combinations of MIN_TOUCHES points
    from itertools import combinations
    for combo_indices in combinations(range(len(significant_points_df)), MIN_TOUCHES):
        combo_points = significant_points_df.iloc[list(combo_indices)]

        # Try all pairs within the combination to form a base line
        for i in range(len(combo_points)):
            for j in range(i + 1, len(combo_points)):
                p1 = combo_points.iloc[i]
                p2 = combo_points.iloc[j]

                p1_x = x_axis[df.index.get_loc(p1.name)]
                p2_x = x_axis[df.index.get_loc(p2.name)]

                if p1_x == p2_x: continue

                p1_y = p1['Low'] if is_support else p1['High']
                p2_y = p2['Low'] if is_support else p2['High']

                slope, intercept = _get_line_equation(p1_x, p1_y, p2_x, p2_y)
                if slope is None: continue

                current_line_coeffs = np.polyfit([p1_x, p2_x], [p1_y, p2_y], 1)
                current_line = np.poly1d(current_line_coeffs)

                # Validate and score this line
                touches = 0
                crossings = 0
                line_start_x = min(p1_x, p2_x)
                line_end_x = max(p1_x, p2_x)

                # Check for touches among all significant points
                touched_points_x = []
                touched_points_y = []
                for _, sp in significant_points_df.iterrows():
                    sp_x = x_axis[df.index.get_loc(sp.name)]
                    sp_y = sp['Low'] if is_support else sp['High']
                    line_y_at_sp_x = current_line(sp_x)
                    
                    # Define tolerance dynamically based on ATR
                    tolerance = df['ATR'].mean() * atr_multiplier
                    
                    if abs(line_y_at_sp_x - sp_y) <= tolerance:
                        touches += 1
                        touched_points_x.append(sp_x)
                        touched_points_y.append(sp_y)

                # Check for crossings over the entire DataFrame
                for k in range(len(df)):
                    current_price_close = df['Close'].iloc[k]
                    line_price_at_k = current_line(x_axis[k])

                    if is_support:
                        # For support, price closing below the line is a crossing
                        if current_price_close < line_price_at_k:
                            crossings += 1
                    else:
                        # For resistance, price closing above the line is a crossing
                        if current_price_close > line_price_at_k:
                            crossings += 1
                
                # Score calculation
                if touches >= MIN_TOUCHES and crossings <= MAX_CROSSINGS:
                    # R-squared calculation for the touched points
                    if len(touched_points_x) > 1:
                        y_predicted = current_line(np.array(touched_points_x))
                        y_actual = np.array(touched_points_y)
                        correlation_matrix = np.corrcoef(y_predicted, y_actual)
                        correlation_xy = correlation_matrix[0,1]
                        r_squared = correlation_xy**2
                    else:
                        r_squared = 0

                    if touches > best_touches:
                        best_touches = touches
                        best_r_squared = r_squared
                        best_line_coeffs = current_line_coeffs
                        best_line_points = combo_points
                    elif touches == best_touches:
                        if r_squared > best_r_squared:
                            best_r_squared = r_squared
                            best_line_coeffs = current_line_coeffs
                            best_line_points = combo_points
    if best_line_coeffs is not None:
        return np.poly1d(best_line_coeffs), best_line_points
    return None, None


# --- Main Logic ---
symbols_to_analyze = []

print("--- Step 1: Finding ATM Options for Indexes ---")
for ticker in tickers:
    if ticker.endswith("-INDEX"):
        symbols_to_analyze.append({'symbol': ticker, 'expiry': None})
        print(f"Queued index for analysis: {ticker}")
        index_name_parts = ticker.split(':')
        index_name = index_name_parts[1].replace('-INDEX', '') if len(index_name_parts) > 1 else ''

        option_index_map = {'NIFTY50': 'NIFTY 50', 'NIFTYBANK': 'NIFTY BANK', 'FINNIFTY': 'NIFTY FIN SERVICE', 'SENSEX': 'SENSEX'}
        index_for_options = option_index_map.get(index_name, index_name)

        quote_data = {"symbols": ticker}
        quote_response = fyers.quotes(data=quote_data)
        if quote_response.get("s") == "ok" and quote_response.get("d"):
            quote_details = quote_response["d"][0]["v"]
            last_price = quote_details.get('lp', quote_details.get('c'))
            if not last_price:
                print(f"\nCould not determine price for {ticker}. Skipping.")
                continue
            print(f"\nProcessing Index: {ticker} | Last Price: {last_price}")

            strike_interval = 100
            if index_for_options == "NIFTY 50": strike_interval = 50

            atm_strike = get_atm_strike(last_price, strike_interval)
            print(f"ATM Strike: {atm_strike}")

            # --- Get Expiry Information based on Index ---
            # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4

            if index_for_options == "NIFTY BANK":
                # Nifty Bank: Monthly expiry on the last Thursday of the month
                monthly_expiry_date = get_monthly_expiry(datetime.now().year, datetime.now().month)
                print(f"Found Monthly Expiry for NIFTY BANK: {monthly_expiry_date.strftime('%Y-%m-%d')}")
                monthly_call = get_fyers_option_symbol(index_for_options, monthly_expiry_date, atm_strike, 'CE', 'MONTHLY')
                monthly_put = get_fyers_option_symbol(index_for_options, monthly_expiry_date, atm_strike, 'PE', 'MONTHLY')
                if monthly_call: symbols_to_analyze.append({'symbol': monthly_call, 'expiry': monthly_expiry_date})
                if monthly_put: symbols_to_analyze.append({'symbol': monthly_put, 'expiry': monthly_expiry_date})
            else:
                # Nifty & Sensex: Weekly and Monthly expiries
                if index_for_options == "NIFTY 50":
                    expiry_day_of_week = 1 # Tuesday for Nifty
                elif index_for_options == "SENSEX":
                    expiry_day_of_week = 3 # Thursday for Sensex
                else: # Default to Thursday
                    expiry_day_of_week = 3

                # 1. Get current week's weekly expiry
                weekly_expiry_date = get_next_weekly_expiry(expiry_day_of_week)
                print(f"Found Weekly Expiry: {weekly_expiry_date.strftime('%Y-%m-%d')}")
                weekly_call = get_fyers_option_symbol(index_for_options, weekly_expiry_date, atm_strike, 'CE', 'WEEKLY')
                weekly_put = get_fyers_option_symbol(index_for_options, weekly_expiry_date, atm_strike, 'PE', 'WEEKLY')
                if weekly_call: symbols_to_analyze.append({'symbol': weekly_call, 'expiry': weekly_expiry_date})
                if weekly_put: symbols_to_analyze.append({'symbol': weekly_put, 'expiry': weekly_expiry_date})

                # 2. Get current month's monthly expiry
                monthly_expiry_date = get_monthly_expiry(datetime.now().year, datetime.now().month)

                # 3. Add monthly options ONLY if they are different from the weekly expiry
                if weekly_expiry_date.date() != monthly_expiry_date.date():
                    print(f"Found Monthly Expiry: {monthly_expiry_date.strftime('%Y-%m-%d')}")
                    monthly_call = get_fyers_option_symbol(index_for_options, monthly_expiry_date, atm_strike, 'CE', 'MONTHLY')
                    monthly_put = get_fyers_option_symbol(index_for_options, monthly_expiry_date, atm_strike, 'PE', 'MONTHLY')
                    if monthly_call: symbols_to_analyze.append({'symbol': monthly_call, 'expiry': monthly_expiry_date})
                    if monthly_put: symbols_to_analyze.append({'symbol': monthly_put, 'expiry': monthly_expiry_date})
                else:
                    print("This week's expiry is also the monthly expiry. Skipping duplicate.")

        else:
            print(f"Could not fetch live price for {ticker}. Error: {quote_response.get('message', 'Unknown error')}")
    elif ticker.endswith("-EQ"):
        symbols_to_analyze.append({'symbol': ticker, 'expiry': None})
        print(f"Queued stock for analysis: {ticker}")
    elif ticker.endswith("-EQ"):
        symbols_to_analyze.append({'symbol': ticker, 'expiry': None})
        print(f"Queued stock for analysis: {ticker}")

print(f"\n--- Step 2: Analyzing {len(symbols_to_analyze)} Symbols ---")
if not os.path.exists('charts'): os.makedirs('charts')

trendlines = {}

for item in symbols_to_analyze:
    symbol = item['symbol']
    expiry_date = item['expiry']

    print(f"\nProcessing {symbol}...")

    days_to_check = DAYS_BACK

    range_to_dt = datetime.combine(datetime.now().date(), dt_time.max)
    range_from_dt = datetime.combine(range_to_dt.date() - timedelta(days=days_to_check), dt_time.min)

    data = {"symbol": symbol, "resolution": INTERVAL, "date_format": "0",
            "range_from": int(range_from_dt.timestamp()),
            "range_to": int(range_to_dt.timestamp()), "cont_flag": "1"}

    response = fyers.history(data=data)

    if response.get("s") != "ok" or not response.get("candles"):
        print(f"No data for {symbol}, skipping. Reason: {response.get('message', 'No candles in response')}")
        time.sleep(0.5)
        continue

    df = pd.DataFrame(response['candles'])
    df.rename(columns={0: 'Datetime', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Volume'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df.set_index('Datetime', inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    df.sort_index(inplace=True)

    df = df[~df.index.duplicated(keep='first')]

    df = calculate_atr(df)

    if df.empty:
        print(f"DataFrame is empty for {symbol} after processing. Skipping.")
        continue

    # Apply CHART_CANDLES limit if set for analysis
    chart_candles_str = os.getenv("CHART_CANDLES")
    if chart_candles_str and chart_candles_str.isdigit():
        chart_candles = int(chart_candles_str)
        if chart_candles > 0 and chart_candles < len(df):
            df_analysis = df.tail(chart_candles).copy()
            print(f"Analyzing the last {chart_candles} candles.")
        else:
            df_analysis = df.copy()
    else:
        df_analysis = df.copy()

    x_axis = np.arange(len(df_analysis.index))

    best_support_line, _ = find_trendlines(df_analysis, is_support=True, atr_multiplier=ATR_MULTIPLIER)
    best_resistance_line, _ = find_trendlines(df_analysis, is_support=False, atr_multiplier=ATR_MULTIPLIER)

    chart_filename = None
    if best_support_line or best_resistance_line:
        clean_symbol_name = symbol.replace(":", "_").replace("-", "_")
        chart_filename = f'charts/{clean_symbol_name}_chart.png'
        
        df_chart = df_analysis.copy()

        # Recalculate x_axis for the potentially smaller df_chart
        x_axis_chart = np.arange(len(df_chart.index))

        aps = []
        if best_support_line:
            # The trendline is already calculated on the df_chart, so no need to adjust
            aps.append(mpf.make_addplot(best_support_line(x_axis_chart), color='g', linestyle='--'))

        if best_resistance_line:
            # The trendline is already calculated on the df_chart, so no need to adjust
            aps.append(mpf.make_addplot(best_resistance_line(x_axis_chart), color='r', linestyle='--'))

        price_min, price_max = df_chart['Low'].min(), df_chart['High'].max()
        y_buffer = (price_max - price_min) * 0.1
        ylim = (price_min - y_buffer, price_max + y_buffer)

        mpf.plot(df_chart, type='candle', style='yahoo', title=f'{symbol} {INTERVAL}-Min Chart', ylabel='Price (INR)',
                 addplot=aps, savefig=chart_filename, ylim=ylim, figsize=(12, 8), tight_layout=True)
        print(f"Chart saved to {chart_filename}")
    else:
        print(f"No valid trendlines found for {symbol}, skipping chart generation.")

    if best_support_line or best_resistance_line:
        trendlines[symbol] = {
            'support': best_support_line,
            'resistance': best_resistance_line,
            'df_length': len(df_analysis)
        }

    if TELEGRAM_ENABLED and (best_support_line or best_resistance_line):
        print("Checking for price alerts...")
        quote_data = {"symbols": symbol}
        quote_response = fyers.quotes(data=quote_data)
        if quote_response.get("s") == "ok" and quote_response.get("d"):
            quote_details = quote_response["d"][0]["v"]
            last_price = quote_details.get('lp', quote_details.get('c'))
            if not last_price:
                print(f"Could not get alert price for {symbol}")
            else:
                x_projected = len(df.index)
                alert_triggered, message = False, ""

                if best_support_line:
                    support_value = best_support_line(x_projected)
                    threshold = support_value * (ALERT_THRESHOLD_PERCENT / 100)
                    if support_value <= last_price <= support_value + threshold:
                        message = (f"📈 *BUY ALERT: {symbol}*\n\nPrice is approaching the support trendline.\n\nCurrent Price: *₹{last_price:.2f}*\nTrendline Price: *₹{support_value:.2f}*")
                        alert_triggered = True

                if not alert_triggered and best_resistance_line:
                    resistance_value = best_resistance_line(x_projected)
                    threshold = resistance_value * (ALERT_THRESHOLD_PERCENT / 100)
                    if resistance_value - threshold <= last_price <= resistance_value:
                        message = (f"📉 *SELL ALERT: {symbol}*\n\nPrice is approaching the resistance trendline.\n\nCurrent Price: *₹{last_price:.2f}*\nTrendline Price: *₹{resistance_value:.2f}*")
                        alert_triggered = True

                if alert_triggered:
                    send_telegram_alert(message, image_path=chart_filename)
        else:
            print("Could not fetch latest price for alert check.")

    time.sleep(1)

print("\nAnalysis complete. ✨")

if trendlines:
    print("\n--- Starting Live Monitoring ---")
    websocket_client = FyersWebsocketClient(symbols=list(trendlines.keys()), trendlines=trendlines)
    websocket_client.start()

# --- Start WebSocket Client for Live Monitoring ---

trendlines = {}
for item in symbols_to_analyze:
    symbol = item['symbol']
    if symbol in df_analysis.columns:
        trendlines[symbol] = {
            'support': best_support_line,
            'resistance': best_resistance_line
        }

if trendlines:
    websocket_client = FyersWebsocketClient(symbols=list(trendlines.keys()), trendlines=trendlines)
    websocket_client.start()