import pandas as pd
import mplfinance as mpf
from fyers_apiv3 import fyersModel
from scipy.signal import find_peaks
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import os
import time
from dotenv import load_dotenv
import requests
import calendar

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

def get_next_weekly_expiry(expiry_weekday):
    today = datetime.now()
    days_ahead = expiry_weekday - today.weekday()
    if days_ahead < 0 or (days_ahead == 0 and today.hour >= 16):
        days_ahead += 7
    return today + timedelta(days=days_ahead)

def get_monthly_expiry(year, month):
    _, last_day = calendar.monthrange(year, month)
    last_date = datetime(year, month, last_day)
    offset = (last_date.weekday() - 3) % 7 # 3 is for Thursday
    return last_date - timedelta(days=offset)

def get_fyers_option_symbol(index_name, expiry_date, strike, option_type, expiry_type):
    expiry_yy = expiry_date.strftime('%y')

    if expiry_type == "WEEKLY":
        expiry_mon = str(expiry_date.month)
        expiry_dd = expiry_date.strftime('%d')
    else: # MONTHLY
        expiry_mon = expiry_date.strftime('%b').upper()
        expiry_dd = ""

    symbol_map = { 'NIFTY 50': 'NIFTY', 'NIFTY BANK': 'BANKNIFTY', 'NIFTY FIN SERVICE': 'FINNIFTY', 'SENSEX': 'SENSEX' }
    base_symbol = symbol_map.get(index_name)
    exchange = "BSE" if index_name == "SENSEX" else "NSE"
    if not base_symbol: return None
    return f"{exchange}:{base_symbol}{expiry_yy}{expiry_mon}{expiry_dd}{strike}{option_type.upper()}"

def find_final_trendline(points, prominences, full_df, x_axis, is_support):
    if len(points) < 2 or prominences is None or len(prominences) < 2: return None
    top_indices = np.argsort(prominences)[-10:]
    significant_points = points.iloc[top_indices]
    if len(significant_points) < 2: return None

    best_line, best_score = None, -1e10
    for i in range(len(significant_points)):
        for j in range(i + 1, len(significant_points)):
            p1, p2 = significant_points.iloc[i], significant_points.iloc[j]
            p1_x = x_axis[full_df.index.get_loc(p1.name)]
            p2_x = x_axis[full_df.index.get_loc(p2.name)]

            if p1_x == p2_x: continue

            y_values = [p1['Low'] if is_support else p1['High'], p2['Low'] if is_support else p2['High']]
            coeffs = np.polyfit([p1_x, p2_x], y_values, 1)
            line = np.poly1d(coeffs)

            crossings, touches = 0, 0
            for k in range(len(full_df)):
                price_at_k = line(x_axis[k])
                if (is_support and full_df['Close'].iloc[k] < price_at_k) or \
                        (not is_support and full_df['Close'].iloc[k] > price_at_k):
                    crossings += 1

            for k in range(len(significant_points)):
                point_k = significant_points.iloc[k]
                point_k_x = x_axis[full_df.index.get_loc(point_k.name)]
                point_k_y = float(point_k['Low'] if is_support else point_k['High'])
                tolerance = 0.5 if point_k_y < 50 else 0.015 * point_k_y
                if abs(line(point_k_x) - point_k_y) < tolerance:
                    touches += 1

            score = touches * 10 - crossings * 1
            if score > best_score:
                best_score, best_line = score, line

    return best_line

# --- Main Logic ---
symbols_to_analyze = []

print("--- Step 1: Finding ATM Options for Indexes ---")
for ticker in tickers:
    if ticker.endswith("-INDEX"):
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

            # --- Get both weekly and monthly expiries ---

            # 1. Get current week's weekly expiry
            # Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4
            expiry_day_of_week = 3 # Thursday default for Nifty
            if index_for_options == "NIFTY BANK":
                expiry_day_of_week = 2 # CORRECTED: Wednesday for Banknifty
            elif index_for_options == "SENSEX":
                expiry_day_of_week = 1 # CORRECTED: Tuesday for Sensex

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

print(f"\n--- Step 2: Analyzing {len(symbols_to_analyze)} Symbols ---")
if not os.path.exists('charts'): os.makedirs('charts')

for item in symbols_to_analyze:
    symbol = item['symbol']
    expiry_date = item['expiry']

    print(f"\nProcessing {symbol}...")

    days_to_check = DAYS_BACK
    if expiry_date:
        today = datetime.now()
        expiry_date = expiry_date.replace(tzinfo=today.tzinfo)
        days_until_expiry = (expiry_date - today).days

        if days_until_expiry >= 5:
            days_to_check = 3
            print(f"Expiry is {days_until_expiry} days away. Using safe lookback of {days_to_check} days.")
        else:
            print(f"Expiry is soon ({days_until_expiry} days). Using full lookback of {days_to_check} days.")

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

    if df.empty:
        print(f"DataFrame is empty for {symbol} after processing. Skipping.")
        continue

    prominence_value = 1 if 'CE' in symbol or 'PE' in symbol else 10
    high_peaks_indices, high_prominences = find_peaks(np.squeeze(df['High'].values), prominence=prominence_value)
    low_peaks_indices, low_prominences = find_peaks(np.squeeze(-df['Low'].values), prominence=prominence_value)

    support_points = df.iloc[low_peaks_indices]
    resistance_points = df.iloc[high_peaks_indices]

    x_axis = np.arange(len(df.index))
    best_support_line = find_final_trendline(support_points, low_prominences.get('prominences'), df, x_axis, is_support=True)
    best_resistance_line = find_final_trendline(resistance_points, high_prominences.get('prominences'), df, x_axis, is_support=False)

    chart_filename = None
    if best_support_line or best_resistance_line:
        clean_symbol_name = symbol.replace(":", "_").replace("-", "_")
        chart_filename = f'charts/{clean_symbol_name}_chart.png'
        aps = []
        if best_support_line: aps.append(mpf.make_addplot(best_support_line(x_axis), color='g', linestyle='--'))
        if best_resistance_line: aps.append(mpf.make_addplot(best_resistance_line(x_axis), color='r', linestyle='--'))

        price_min, price_max = df['Low'].min(), df['High'].max()
        y_buffer = (price_max - price_min) * 0.1
        ylim = (price_min - y_buffer, price_max + y_buffer)

        mpf.plot(df, type='candle', style='yahoo', title=f'{symbol} {INTERVAL}-Min Chart', ylabel='Price (INR)',
                 addplot=aps, savefig=chart_filename, ylim=ylim, figsize=(12, 8), tight_layout=True)
        print(f"Chart saved to {chart_filename}")
    else:
        print(f"No valid trendlines found for {symbol}, skipping chart generation.")

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