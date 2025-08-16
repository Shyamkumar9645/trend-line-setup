import pandas as pd
import yfinance as yf
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np

# List of Indian stocks
tickers = [
    'ADANIENT.NS',      # Adani Enterprises
    'ADANIPORTS.NS',    # Adani Ports & SEZ
    'ASIANPAINT.NS',    # Asian Paints
    'AXISBANK.NS',      # Axis Bank
    'BAJAJ-AUTO.NS',    # Bajaj Auto
    'BAJFINANCE.NS',    # Bajaj Finance
    'BAJAJFINSV.NS',    # Bajaj Finserv
    'BPCL.NS',          # Bharat Petroleum Corp
    'BHARTIARTL.NS',    # Bharti Airtel
    'BRITANNIA.NS',     # Britannia Industries
    'CIPLA.NS',         # Cipla
    'COALINDIA.NS',     # Coal India
    'DIVISLAB.NS',      # Divi’s Laboratories
    'DRREDDY.NS',       # Dr. Reddy’s Laboratories
    'EICHERMOT.NS',     # Eicher Motors
    'GRASIM.NS',        # Grasim Industries
    'HCLTECH.NS',       # HCL Technologies
    'HDFCBANK.NS',      # HDFC Bank
    'HDFC.NS',          # Housing Development Finance Corp
    'HDFCLIFE.NS',      # HDFC Life Insurance
    'HEROMOTOCO.NS',    # Hero MotoCorp
    'HINDALCO.NS',      # Hindalco Industries
    'HINDUNILVR.NS',    # Hindustan Unilever
    'ICICIBANK.NS',     # ICICI Bank
    'ITC.NS',           # ITC
    'INDUSINDBK.NS',    # IndusInd Bank
    'INFY.NS',          # Infosys
    'JSWSTEEL.NS',      # JSW Steel
    'KOTAKBANK.NS',     # Kotak Mahindra Bank
    'LT.NS',            # Larsen & Toubro
    'M&M.NS',           # Mahindra & Mahindra
    'MARUTI.NS',        # Maruti Suzuki
    'NTPC.NS',          # NTPC
    'NESTLEIND.NS',     # Nestle India
    'ONGC.NS',          # Oil & Natural Gas Corp
    'PIDILITIND.NS',    # Pidilite Industries
    'POWERGRID.NS',     # Power Grid Corporation
    'RELIANCE.NS',      # Reliance Industries
    'SBIN.NS',          # State Bank of India
    'SUNPHARMA.NS',     # Sun Pharmaceutical
    'TCS.NS',           # Tata Consultancy Services
    'TATACONSUM.NS',    # Tata Consumer Products
    'TATAMOTORS.NS',    # Tata Motors
    'TATASTEEL.NS',     # Tata Steel
    'TECHM.NS',         # Tech Mahindra
    'TITAN.NS',         # Titan Company
    'ULTRACEMCO.NS',    # UltraTech Cement
    'UPL.NS',           # UPL Ltd
    'WIPRO.NS'          # Wipro
]

for ticker in tickers:
    print(f"Processing {ticker}...")
    # Fetch stock data
    df = yf.download(ticker, period='2d', interval='1m')

    if df.empty:
        print(f"No data found for {ticker}, skipping.")
        continue

    # Clean up the column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Find peaks (resistance)
    high_peaks_indices, _ = find_peaks(np.squeeze(df['High'].values), prominence=10)

    # Find troughs (support)
    low_peaks_indices, _ = find_peaks(np.squeeze(-df['Low'].values), prominence=10)

    def find_final_trendline(points, prominences, full_df, x_axis, is_support):
        if len(points) < 2:
            return None

        # Filter for the top 10 most prominent swings
        top_indices = np.argsort(prominences)[-10:]
        significant_points = points.iloc[top_indices]

        if len(significant_points) < 2:
            return None

        best_line = None
        best_score = -1e10

        for i in range(len(significant_points)):
            for j in range(i + 1, len(significant_points)):
                p1 = significant_points.iloc[i]
                p2 = significant_points.iloc[j]

                # Create a candidate line
                p1_x = x_axis[df.index.get_loc(p1.name)]
                p2_x = x_axis[df.index.get_loc(p2.name)]
                y_values = [p1['Low'] if is_support else p1['High'], p2['Low'] if is_support else p2['High']]
                coeffs = np.polyfit([p1_x, p2_x], y_values, 1)
                line = np.poly1d(coeffs)

                # Score the line
                score = 0
                crossings = 0
                touches = 0

                # Penalize for crossings
                for k in range(len(full_df)):
                    price_at_k = line(x_axis[k])
                    if is_support and full_df['Close'].iloc[k] < price_at_k:
                        crossings += 1
                    elif not is_support and full_df['Close'].iloc[k] > price_at_k:
                        crossings += 1

                # Reward for touches
                for k in range(len(significant_points)):
                    point_k = significant_points.iloc[k]
                    point_k_x = x_axis[df.index.get_loc(point_k.name)]
                    point_k_y = point_k['Low'] if is_support else point_k['High']
                    if abs(line(point_k_x) - point_k_y) < 0.015 * point_k_y: # 1.5% tolerance
                        touches += 1

                score = touches * 10 - crossings * 1

                if score > best_score:
                    best_score = score
                    best_line = line

        return best_line

    # Prepare trendlines
    support_points = df.iloc[low_peaks_indices]
    resistance_points = df.iloc[high_peaks_indices]

    # Get prominences of the peaks
    _, low_prominences = find_peaks(-df['Low'].values, prominence=10)
    _, high_prominences = find_peaks(df['High'].values, prominence=10)

    # Create a numerical index for the x-axis
    x_axis = np.arange(len(df.index))

    best_support_line = find_final_trendline(support_points, low_prominences['prominences'], df, x_axis, is_support=True)
    best_resistance_line = find_final_trendline(resistance_points, high_prominences['prominences'], df, x_axis, is_support=False)

    support_trend = best_support_line(x_axis) if best_support_line else None
    resistance_trend = best_resistance_line(x_axis) if best_resistance_line else None

    # Skip chart generation if no trendlines were found
    if support_trend is None and resistance_trend is None:
        print(f"No valid trendlines found for {ticker}, skipping chart generation.")
        continue

    # Create a list of additional plots
    aps = []
    if support_trend is not None:
        aps.append(mpf.make_addplot(support_trend, color='g', linestyle='--'))
    if resistance_trend is not None:
        aps.append(mpf.make_addplot(resistance_trend, color='r', linestyle='--'))

    # Set y-axis limits to focus on the price action
    price_min = df['Low'].min()
    price_max = df['High'].max()
    y_buffer = (price_max - price_min) * 0.1 # 10% buffer
    ylim = (price_min - y_buffer, price_max + y_buffer)

    # Create the candlestick chart
    chart_filename = f'charts/{ticker}_candlestick_chart.png'
    mpf.plot(df, type='candle', style='yahoo',
             title=f'{ticker} Stock Price',
             ylabel='Price (INR)',
             addplot=aps,
             savefig=chart_filename,
             ylim=ylim)

    print(f"Chart saved to {chart_filename}")