from fyers_apiv3.FyersWebsocket import data_ws
import os
from dotenv import load_dotenv
import requests

load_dotenv(override=True)

CLIENT_ID = os.getenv("CLIENT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

def send_telegram_alert(message):
    if not TELEGRAM_ENABLED: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, json=payload)
        if response.status_code == 200: print("✅ Telegram alert sent successfully.")
        else: print(f"❌ Failed to send Telegram alert: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ An error occurred while sending Telegram alert: {e}")

class FyersWebsocketClient:
    def __init__(self, symbols, trendlines):
        self.symbols = symbols
        self.trendlines = trendlines
        self.access_token = f"{CLIENT_ID}:{ACCESS_TOKEN}"
        self.fyers = data_ws.FyersDataSocket(
            access_token=self.access_token,
            on_connect=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
        )

    def on_message(self, message):
        if isinstance(message, list) and len(message) > 0 and 'symbol' in message[0] and 'ltp' in message[0]:
            message = message[0]
            symbol = message['symbol']
            ltp = message['ltp']

            if symbol in self.trendlines:
                trendline = self.trendlines[symbol]
                support_line = trendline['support']
                resistance_line = trendline['resistance']
                df_length = trendline['df_length']

                threshold = 50 if symbol.endswith("-INDEX") else 10

                if support_line:
                    support_value = support_line(df_length) # Project to the next candle
                    if support_value <= ltp <= support_value + threshold:
                        message_text = (f"📈 *BUY ALERT: {symbol}*\n\nPrice is approaching the support trendline.\n\nCurrent Price: *₹{ltp:.2f}*\nTrendline Price: *₹{support_value:.2f}*")
                        send_telegram_alert(message_text)

                if resistance_line:
                    resistance_value = resistance_line(df_length) # Project to the next candle
                    if resistance_value - threshold <= ltp <= resistance_value:
                        message_text = (f"📉 *SELL ALERT: {symbol}*\n\nPrice is approaching the resistance trendline.\n\nCurrent Price: *₹{ltp:.2f}*\nTrendline Price: *₹{resistance_value:.2f}*")
                        send_telegram_alert(message_text)

    def on_error(self, message):
        print("Error:", message)

    def on_close(self, message):
        print("Connection closed:", message)

    def on_open(self):
        data_type = "SymbolUpdate"
        self.fyers.subscribe(symbols=self.symbols, data_type=data_type)
        self.fyers.keep_running()

    def start(self):
        self.fyers.connect()
