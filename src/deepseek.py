import pandas as pd
import mplfinance as mpf
import numpy as np
import os
import time
from dotenv import load_dotenv
import requests
import calendar
import threading
import json
import logging
from datetime import datetime, timedelta, time as dt_time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import argrelextrema
import matplotlib
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

matplotlib.use('Agg')

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
INTERVAL = os.getenv("INTERVAL", "5")
DAYS_BACK = int(os.getenv("DAYS_BACK", 1))
CHART_CANDLES = os.getenv("CHART_CANDLES")
CHART_CANDLES = int(CHART_CANDLES) if CHART_CANDLES and CHART_CANDLES.isdigit() and int(CHART_CANDLES)>0 else None
LOOKBACK = int(os.getenv("LOOKBACK", 5))
LOOKAHEAD = int(os.getenv("LOOKAHEAD", 5))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", 1.0))
MIN_SWING_MULTIPLIER = float(os.getenv("MIN_SWING_MULTIPLIER", 1.2))
MIN_POINTS = int(os.getenv("MIN_POINTS", 2))
MAX_CROSSINGS = int(os.getenv("MAX_CROSSINGS", 2))
MIN_TOUCHES = int(os.getenv("MIN_TOUCHES", 3))
MIN_R_SQUARED = float(os.getenv("MIN_R_SQUARED", 0.9))
ALERT_OPTION_THRESHOLD = float(os.getenv("ALERT_OPTION_THRESHOLD", 50))
ALERT_STOCK_THRESHOLD = float(os.getenv("ALERT_STOCK_THRESHOLD", 50))
ALERT_COOLDOWN_MINUTES = float(os.getenv("ALERT_COOLDOWN_MINUTES", 60))
RUN_INTERVAL_MINUTES = int(os.getenv("RUN_INTERVAL_MINUTES", 5))
DEBUG_ALERTS = os.getenv("DEBUG_ALERTS", "False").lower() == "true"

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
tickers = list(dict.fromkeys(base_indexes + env_tickers))

# --- Fyers Model Initialization ---
try:
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, log_path=os.getcwd())
except Exception as e:
    logger.error(f"❌ Error initializing FyersModel: {e}")
    exit()

trendlines = {}
last_alerts = {}
first_run = True

# --- Helper Functions ---
def send_telegram_alert(message, image_path=None):
    """Send alert to Telegram with optional image attachment"""
    if not TELEGRAM_ENABLED: return
    try:
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
                requests.post(url, data=data, files=files, timeout=30)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
            requests.post(url, json=payload, timeout=30)
    except Exception as e:
        logger.error(f"❌ Error sending Telegram alert: {e}")

def get_atm_strike(spot_price, strike_interval=50):
    return int(round(spot_price/strike_interval)*strike_interval)

def get_next_weekly_expiry(expiry_weekday):
    today = datetime.now()
    current_weekday = today.weekday()
    days_ahead = expiry_weekday - current_weekday
    if days_ahead <= 0: days_ahead += 7
    expiry_date = today + timedelta(days=days_ahead)
    if expiry_date.date()==today.date() and today.hour>=15 and today.minute>=30: expiry_date += timedelta(days=7)
    return expiry_date

def get_monthly_expiry(year, month, expiry_weekday):
    _, last_day = calendar.monthrange(year, month)
    last_date = datetime(year, month, last_day)
    offset = (last_date.weekday()-expiry_weekday)%7
    expiry_date = last_date-timedelta(days=offset)
    today = datetime.now()
    if expiry_date.date()<today.date() or (expiry_date.date()==today.date() and today.hour>=15 and today.minute>=30):
        if month==12: year+=1; month=1
        else: month+=1
        _, last_day = calendar.monthrange(year, month)
        last_date = datetime(year,month,last_day)
        offset = (last_date.weekday()-expiry_weekday)%7
        expiry_date = last_date-timedelta(days=offset)
    return expiry_date

def get_fyers_option_symbol(index_name, expiry_date, strike, option_type, expiry_type):
    expiry_yy = expiry_date.strftime('%y')
    expiry_mon = expiry_date.strftime('%b').upper()
    expiry_m = str(int(expiry_date.strftime('%m')))
    expiry_dd = expiry_date.strftime('%d')
    symbol_map = {
        'NIFTY 50': 'NIFTY',
        'NIFTY BANK': 'BANKNIFTY',
        'NIFTY FIN SERVICE': 'FINNIFTY',
        'SENSEX': 'SENSEX'
    }
    base_symbol = symbol_map.get(index_name)
    exchange = "BSE" if index_name=="SENSEX" else "NSE"
    if not base_symbol: return None
    if index_name=="SENSEX":
        if expiry_type=="WEEKLY":
            symbol = f"{exchange}:SENSEX{expiry_yy}{expiry_m}{expiry_dd}{strike}{option_type.upper()}"
        else:
            symbol = f"{exchange}:SENSEX{expiry_yy}{expiry_mon}{strike}{option_type.upper()}"
    else:
        if expiry_type=="WEEKLY":
            symbol = f"{exchange}:{base_symbol}{expiry_yy}{expiry_m}{expiry_dd}{strike}{option_type.upper()}"
        else:
            symbol = f"{exchange}:{base_symbol}{expiry_yy}{expiry_mon}{strike}{option_type.upper()}"
    return symbol

def calculate_atr(df, period=14):
    high_low = df["High"]-df["Low"]
    high_close = np.abs(df["High"]-df["Close"].shift())
    low_close = np.abs(df["Low"]-df["Close"].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

def find_pivot_points(df, lookback, lookahead, min_amplitude=0):
    if len(df)<lookback+lookahead+1: return pd.DataFrame(), pd.DataFrame()
    order = max(3,min(5,len(df)//20))
    maxes = argrelextrema(df["High"].values,np.greater,order=order)[0]
    highs = [dict(Datetime=df.index[i],High=df['High'].iloc[i],Low=df['Low'].iloc[i],Close=df['Close'].iloc[i])
             for i in maxes if i>=lookback and i<len(df)-lookahead
             and (df['High'].iloc[i]-df['Low'].iloc[max(0,i-lookback):i].min())>=min_amplitude
             and (df['High'].iloc[i]-df['Low'].iloc[i:i+lookahead].min())>=min_amplitude*0.5]
    mins = argrelextrema(df["Low"].values,np.less,order=order)[0]
    lows = [dict(Datetime=df.index[i],Low=df['Low'].iloc[i],High=df['High'].iloc[i],Close=df['Close'].iloc[i])
            for i in mins if i>=lookback and i<len(df)-lookahead
            and (df['High'].iloc[max(0,i-lookback):i].max()-df['Low'].iloc[i])>=min_amplitude
            and (df['High'].iloc[i:i+lookahead].max()-df['Low'].iloc[i])>=min_amplitude*0.5]
    dfh = pd.DataFrame(highs).set_index("Datetime") if highs else pd.DataFrame()
    dfl = pd.DataFrame(lows).set_index("Datetime") if lows else pd.DataFrame()
    return dfh, dfl

def calculate_trendline_quality(line, df, x_axis, is_support, tolerance):
    touches=crossings=near_touches=price_violations=0; total_distance=0
    for i in range(len(df)):
        trendline_price=line(x_axis[i]); actual=df["Low"].iloc[i] if is_support else df["High"].iloc[i]; close_p=df["Close"].iloc[i]
        distance=abs(actual-trendline_price); total_distance+=distance
        if distance<=tolerance: touches+=1
        elif distance<=tolerance*2: near_touches+=1
        if is_support:
            if close_p<trendline_price-tolerance: crossings+=1; price_violations+=(trendline_price-close_p)/trendline_price
        else:
            if close_p>trendline_price+tolerance: crossings+=1; price_violations+=(close_p-trendline_price)/trendline_price
    avg_distance=total_distance/len(df)
    quality_score=(touches*15)+(near_touches*5)+(1/(avg_distance+1))*20+max(0,(10-crossings))*3+(1/(price_violations+1))*10
    return dict(score=quality_score, touches=touches, crossings=crossings)

def find_best_sloped_trendline(points_df, full_df, is_support, tolerance, atr):
    if points_df is None or len(points_df)<MIN_POINTS: return None,0,None
    y_col="Low" if is_support else "High"; min_slope=atr*0.01; sloped=[]
    for i in range(len(points_df)):
        for j in range(i+1,len(points_df)):
            p1, p2 = points_df.iloc[i], points_df.iloc[j]
            si,ei = full_df.index.get_loc(p1.name), full_df.index.get_loc(p2.name)
            if ei-si<1: continue
            xs,ys = np.array([si,ei]).reshape(-1,1),np.array([p1[y_col],p2[y_col]])
            model = LinearRegression().fit(xs, ys)
            slope=model.coef_[0]
            if abs(slope)<min_slope: continue
            line_poly=np.poly1d([slope,model.intercept_])
            qual=calculate_trendline_quality(line_poly, full_df.iloc[si:ei+1], np.arange(si,ei+1), is_support, tolerance)
            preds=line_poly(np.arange(si,ei+1)); acts=full_df[y_col].iloc[si:ei+1]
            r2=r2_score(acts,preds) if len(acts)>1 else 0
            if qual['touches']>=MIN_TOUCHES and qual['crossings']<=MAX_CROSSINGS and r2>=MIN_R_SQUARED:
                sloped.append({'line':line_poly,'touches':qual['touches'],'range':(si,ei),'score':qual['score']})
    if not sloped: return None,0,None
    best=max(sloped,key=lambda x:x['score']); return best['line'],best['touches'],best['range']

def find_best_horizontal_trendline(points_df, full_df, is_support, tolerance):
    if points_df is None or len(points_df)<MIN_POINTS: return None,0,None
    y_col="Low" if is_support else "High"; ht=tolerance*0.5
    prices,indices=points_df[y_col].values,[full_df.index.get_loc(dt) for dt in points_df.index]; clusters=[]
    for i,price in enumerate(prices):
        cluster_prices=[price]; cluster_indices=[indices[i]]
        for j in range(len(prices)):
            if i!=j and abs(price-prices[j])<=ht: cluster_prices.append(prices[j]); cluster_indices.append(indices[j])
        if len(cluster_prices)>=MIN_TOUCHES: clusters.append({'avg_price':np.mean(cluster_prices),'indices':cluster_indices,'touches':len(cluster_prices)})
    if not clusters: return None,0,None
    lines=[]
    for c in clusters:
        ap,inds = c['avg_price'],sorted(c['indices'])
        si,ei = min(inds),max(inds)
        line_poly = np.poly1d([0,ap])
        qual=calculate_trendline_quality(line_poly, full_df.iloc[si:ei+1],np.arange(si,ei+1),is_support,ht)
        if qual['touches']>=MIN_TOUCHES and qual['crossings']<=MAX_CROSSINGS:
            lines.append({'line':line_poly,'touches':qual['touches'],'range':(si,ei),'score':qual['score']})
    if not lines: return None,0,None
    best=max(lines,key=lambda x:x['score']); return best['line'],best['touches'],best['range']

def find_best_trendline(points_df, full_df, x_axis, is_support, tolerance, atr):
    if points_df is None or len(points_df)<MIN_POINTS: return (None,0,None,None,0,None,None)
    points_df=points_df.sort_index()
    slp,s_t,s_rng=find_best_sloped_trendline(points_df,full_df,is_support,tolerance,atr)
    hor,h_t,h_rng=find_best_horizontal_trendline(points_df,full_df,is_support,tolerance)
    def check_alert(line_poly, rng, is_sup, tol, price, lx, tlnm):
        if line_poly is None or rng is None: return False,None
        si,ei = rng
        if si<=lx<=ei:
            tprice = line_poly(lx); a_tol = tol if tlnm.startswith('Sloped') else tol*0.5
            if abs(price-tprice)<=a_tol:
                return True, {'alert_type': ('Support' if is_sup else 'Resistance'), 'price': price, 'trendline_value': tprice, 'line_type':tlnm}
        return False,None
    return slp,s_t,s_rng,hor,h_t,h_rng,check_alert

def create_trendline_chart(df, symbol, sloped_support=None, sloped_resist=None,
    horizontal_support=None, horizontal_resist=None,
    sloped_support_range=None, sloped_resist_range=None,
    horizontal_support_range=None, horizontal_resist_range=None,
    swing_lows_df=None, swing_highs_df=None):
    if df.empty or len(df)<2: return None
    image_dir='charts'; os.makedirs(image_dir,exist_ok=True)
    chart_file=f'{image_dir}/{symbol.replace(":","_")}_trendlines.png'
    x_axes = np.arange(len(df.index)); aps=[]
    def addtl(l,rng,c,sty,lbl):
        if l is not None and rng is not None:
            vals=np.full(len(x_axes),np.nan); si,ei = rng
            for i in range(max(0,si),min(len(x_axes),ei+1)): vals[i]=l(i)
            aps.append(mpf.make_addplot(vals,color=c,linestyle=sty,width=2,alpha=0.8,label=lbl))
    addtl(sloped_support,sloped_support_range,'green','-','Sloped Support')
    addtl(horizontal_support,horizontal_support_range,'limegreen','--','Horizontal Support')
    addtl(sloped_resist,sloped_resist_range,'red','-','Sloped Resistance')
    addtl(horizontal_resist,horizontal_resist_range,'darkred','--','Horizontal Resistance')
    price_min, price_max = df["Low"].min(), df["High"].max()
    ylim = (price_min-(price_max-price_min)*0.1, price_max+(price_max-price_min)*0.1)
    style = mpf.make_mpf_style(base_mpf_style='yahoo')
    mpf.plot(df,type='candle',addplot=aps if aps else None,style=style, title=f"{symbol} Trendlines",
        ylabel="Price (₹)", savefig=chart_file, ylim=ylim, figsize=(14,8), tight_layout=True, volume=False, show_nontrading=False)
    return chart_file if os.path.exists(chart_file) else None

# --- Main Analysis Function ---
def analyze():
    global trendlines, first_run
    symbols_to_analyze = []
    today = datetime.now(); cy, cm = today.year, today.month
    for ticker in tickers:
        if ticker.endswith("-INDEX"):
            try:
                index_name_parts = ticker.split(':')
                index_name = index_name_parts[1].replace('-INDEX','') if len(index_name_parts)>1 else ''
                option_index_map = {'NIFTY50':'NIFTY 50','NIFTYBANK':'NIFTY BANK','FINNIFTY':'NIFTY FIN SERVICE','SENSEX':'SENSEX'}
                index_for_options = option_index_map.get(index_name,index_name)
                quote_data = {'symbols': ticker}
                quote_response = fyers.quotes(data=quote_data)
                if quote_response.get("s")=="ok" and quote_response.get("d"):
                    ltp = quote_response["d"][0]["v"].get('lp', quote_response["d"][0]["v"].get('c'))
                    if not ltp: logger.warning(f"{ticker} has no price. Skipping."); continue
                    strike_interval = 50 if index_for_options=="NIFTY 50" else 100
                    atm_strike = get_atm_strike(ltp,strike_interval)
                    expiry_weekday = 1 if index_for_options in ["NIFTY 50","NIFTY BANK"] else 3
                    weekly_expiry = get_next_weekly_expiry(expiry_weekday)
                    monthly_expiry = get_monthly_expiry(cy,cm,expiry_weekday)
                    for expiry_date, exp_type in [(weekly_expiry,'WEEKLY'),(monthly_expiry,'MONTHLY')]:
                        if expiry_date and (not weekly_expiry or weekly_expiry.date()!=monthly_expiry.date() or exp_type=='WEEKLY'):
                            for opttype in ['CE','PE']:
                                symbol=get_fyers_option_symbol(index_for_options, expiry_date, atm_strike, opttype, exp_type)
                                if symbol is not None:
                                    symbols_to_analyze.append({'symbol':symbol, 'expiry':expiry_date})
            except Exception as e:
                logger.error(f"XX Error processing {ticker}: {e}")
        elif ticker.endswith("-EQ"):
            symbols_to_analyze.append({'symbol':ticker,'expiry':None})

    trendlines.clear()
    for item in symbols_to_analyze:
        symbol=item['symbol']
        expiry=item['expiry']
        if expiry and expiry.date()>today.date()+timedelta(days=31): continue
        logger.info(f"\nAnalyzing {symbol} ...")
        try:
            range_to = datetime.combine(today.date(),dt_time.max)
            range_from = datetime.combine(range_to.date()-timedelta(days=DAYS_BACK),dt_time.min)
            data = {"symbol":symbol,"resolution":INTERVAL,"date_format":"0",
                    "range_from":int(range_from.timestamp()),"range_to":int(range_to.timestamp()),"cont_flag":"1"}
            response = fyers.history(data=data)
            if response.get("s")!="ok" or not response.get("candles"): continue
            df = pd.DataFrame(response['candles'],columns=['Datetime','Open','High','Low','Close','Volume'])
            df['Datetime'] = pd.to_datetime(df['Datetime'],unit='s')
            df.set_index('Datetime', inplace=True)
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            if df.empty: continue
            if CHART_CANDLES and len(df)>CHART_CANDLES: df = df.tail(CHART_CANDLES)
            atr=calculate_atr(df)
            if atr.dropna().empty: continue
            tolerance=atr.mean()*ATR_MULTIPLIER; min_amplitude=atr.mean()*MIN_SWING_MULTIPLIER
            trendlines[symbol]={'df':df,'tolerance':tolerance}
            swing_highs_df,swing_lows_df=find_pivot_points(df,LOOKBACK,LOOKAHEAD,min_amplitude)
            x_axis = np.arange(len(df.index))
            sl_s,sl_t,sl_rng,hor_s,hor_t,hor_rng,alert_func = (None,0,None,None,0,None,None)
            sl_r,sl_rt,slr_rng,hor_r,hor_rt,horrr,_ = (None,0,None,None,0,None,None)
            if swing_lows_df is not None and not swing_lows_df.empty and len(swing_lows_df)>=MIN_POINTS:
                sl_s,sl_t,sl_rng,hor_s,hor_t,hor_rng,alert_func = find_best_trendline(swing_lows_df,df,x_axis,True,tolerance,atr.mean())
            if swing_highs_df is not None and not swing_highs_df.empty and len(swing_highs_df)>=MIN_POINTS:
                sl_r,sl_rt,slr_rng,hor_r,hor_rt,horrr,_ = find_best_trendline(swing_highs_df,df,x_axis,False,tolerance,atr.mean())
            if (sl_s is not None or hor_s is not None or sl_r is not None or hor_r is not None):
                trendlines[symbol].update({
                    'sloped_support': sl_s, 'sloped_support_range': sl_rng, 'sloped_support_touches': sl_t,
                    'horizontal_support': hor_s, 'horizontal_support_range': hor_rng, 'horizontal_support_touches': hor_t,
                    'sloped_resistance': sl_r, 'sloped_resistance_range': slr_rng, 'sloped_resistance_touches': sl_rt,
                    'horizontal_resistance': hor_r, 'horizontal_resistance_range': horrr, 'horizontal_resistance_touches': hor_rt,
                    'alert_func':alert_func, 'x_axis':x_axis
                })
                chart_file = create_trendline_chart(df,symbol,sl_s,sl_r,hor_s,hor_r,sl_rng,slr_rng,hor_rng,horrr)
                if chart_file: trendlines[symbol]['chart_filename'] = chart_file
        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")

# --- WebSocket ---
def check_and_send_alert(symbol, price, is_initial_alert=False):
    global last_alerts, trendlines
    if symbol not in trendlines: return
    if (not is_initial_alert and symbol in last_alerts and (datetime.now()-last_alerts[symbol]).total_seconds()/60 < ALERT_COOLDOWN_MINUTES): return
    df=trendlines[symbol]['df']; chart_filename=trendlines[symbol].get('chart_filename')
    tolerance=trendlines[symbol].get('tolerance'); alert_func=trendlines[symbol].get('alert_func')
    x_projected=len(df.index)-1; triggered=False; action=""; line_type=""
    for lkey,rngkey,is_s,tln,tch,_ in [
            ('sloped_support','sloped_support_range',True,'Sloped Support','sloped_support_touches',0),
            ('horizontal_support','horizontal_support_range',True,'Horizontal Support','horizontal_support_touches',0),
            ('sloped_resistance','sloped_resistance_range',False,'Sloped Resistance','sloped_resistance_touches',0),
            ('horizontal_resistance','horizontal_resistance_range',False,'Horizontal Resistance','horizontal_resistance_touches',0)]:
        line=trendlines[symbol].get(lkey)
        rng=trendlines[symbol].get(rngkey)
        touches=trendlines[symbol].get(tch,0)
        if line is not None and alert_func and not triggered:
            tr,det=alert_func(line,rng,is_s,tolerance,price,x_projected,tln)
            if tr:
                action="BUY" if is_s else "SELL"; line_type=tln
                alert_title=f"{'📈' if is_s else '📉'} *{action} ALERT: {symbol}*"
                alert_body=f"🎯 *{det['line_type'].upper()} TOUCHED*\n💰 Price: *₹{price:.2f}*\n📊 {det['alert_type']}: *₹{det['trendline_value']:.2f}*\n🎯 Touches: *{touches}*\n🕐 {datetime.now().strftime('%H:%M:%S')}"
                send_telegram_alert(f"{alert_title}\n\n{alert_body}", chart_filename)
                last_alerts[symbol]=datetime.now(); triggered=True

def on_message(message):
    try:
        data=json.loads(message) if isinstance(message,str) else message
        if 'symbol' in data and 'ltp' in data: check_and_send_alert(data['symbol'],data['ltp'])
    except Exception as e: logger.error(f"WebSocket msg error: {e}")

def on_error(error): logger.error(f"WebSocket Error: {error}")
def on_close(msg): logger.info(f"WebSocket Closed: {msg}")
def on_open():
    logger.info("WebSocket Connected."); symbols=list(trendlines.keys())
    if symbols: fyers_ws.subscribe(symbols=symbols,data_type="SymbolUpdate")

def run_websocket():
    global fyers_ws
    try:
        fyers_ws = data_ws.FyersDataSocket(
            access_token=f"{CLIENT_ID}:{ACCESS_TOKEN}",
            log_path=os.getcwd(),
            on_connect=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close)
        fyers_ws.connect()
        logger.info("WebSocket thread started")
    except Exception as e: logger.error(f"WebSocket err: {e}")

def main():
    global first_run
    if TELEGRAM_ENABLED:
        send_telegram_alert("🤖 *Fyers Trendline Bot Started*")
    ws_thread = threading.Thread(target=run_websocket,daemon=True)
    ws_thread.start(); time.sleep(5)
    while True:
        try: analyze(); logger.info(f"Sleeping for {RUN_INTERVAL_MINUTES}m..."); time.sleep(RUN_INTERVAL_MINUTES*60)
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Main error: {e}")
            if TELEGRAM_ENABLED:
                send_telegram_alert(f"❌ ERROR: {str(e)}\nRetry in {RUN_INTERVAL_MINUTES}m..")
            time.sleep(RUN_INTERVAL_MINUTES*60)

if __name__=="__main__":
    main()
