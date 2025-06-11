import numpy as np
import pandas as pd
import MetaTrader5 as mt5

class CurrentRates():
    @staticmethod
    def get_daily_rates(currency):
        daily_rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_D1,0,10000)
        df = pd.DataFrame(daily_rates)
        # df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'])
        daily_df1 = df.set_index('time')
        return df
        
    def get_4hour_rates(currency):
        four_hour_rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_H4,0,10000)
        df = pd.DataFrame(four_hour_rates)
        df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'])
        # four_hour_df1 = df.set_index('time')
        return df
        
    def get_hourly_rates(currency):
        one_hour_rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_H1,0,10000)
        df = pd.DataFrame(one_hour_rates)
        df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        # one_hour_df1 = df.set_index('time')
        return df
    
    def get_minute_rates(currency):
        one_min_rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_M1,0,10000)
        df = pd.DataFrame(one_min_rates)
        df = df.drop(columns=['tick_volume','spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        # print()
        return df
    
    def get_30min_rates(currency):
        rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_M30,0,10000)
        df = pd.DataFrame(rates)
        df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        return df
    
    
    def get_5min_rates(currency):
        rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_M5,0,10000)
        df = pd.DataFrame(rates)
        df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        return df
    
    def get_15min_rates(currency):
        rates = mt5.copy_rates_from_pos(currency,mt5.TIMEFRAME_M15,0,10000)
        df = pd.DataFrame(rates)
        df = df.drop(columns=['spread','real_volume'])
        df['time'] = pd.to_datetime(df['time'],unit='s')
        return df
    
    


class Indicators():
    @staticmethod
    def edit_data(df):
        new_df = df.drop(columns=['spread','real_volume'])

        return new_df
    

    def ichimoku_strategy(df):
        """
        Applies a basic Ichimoku Cloud trading strategy to determine buy and sell signals.

        Parameters:
        df (pd.DataFrame): DataFrame containing OHLC data and Ichimoku indicator columns.

        Returns:
        df (pd.DataFrame): Modified DataFrame with 'Buy_Signal' and 'Sell_Signal' columns.
        """
        # Ensure Ichimoku components are calculated (calling previous function if needed)
        hi_val = df['high'].rolling(window=9).max()
        low_val = df['low'].rolling(window=9).min()
        df['Conversion'] = (hi_val + low_val) / 2

        hi_val2 = df['high'].rolling(window=26).max()
        low_val2 = df['low'].rolling(window=26).min()
        df['Baseline'] = (hi_val2 + low_val2) / 2

        df['SpanA'] = (df['Conversion'] + df['Baseline']) / 2
        hi_val3 = df['high'].rolling(window=52).max()
        low_val3 = df['low'].rolling(window=52).min()
        df['SpanB'] = (hi_val3 + low_val3) / 2
        df['Lagging'] = df['close'].shift(-26)

        # Strategy Conditions
        df['Above_Cloud'] = (df['close'] > df['SpanA']) & (df['close'] > df['SpanB'])
        df['Below_Cloud'] = (df['close'] < df['SpanA']) & (df['close'] < df['SpanB'])

        # Buy and Sell Signals based on Cloud Position
        df['Buy_Signal'] = np.where(df['Above_Cloud'], 1, 0)
        df['Sell_Signal'] = np.where(df['Below_Cloud'], -1, 0)

        return df
    
    def get_fractals(df):
        df['High_Fractal'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1)) &
            (df['high'].shift(1) > df['high'].shift(2)) &
            (df['high'].shift(-1) > df['high'].shift(-2)) 
        )
        df['Low_Fractal'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1)) &
            (df['low'].shift(1) < df['low'].shift(2)) &
            (df['low'].shift(-1) < df['low'].shift(-2)) 
        )
        
        return df
    
    def liquidity_zones(df, threshold=0.5):
        df['Candle_Body'] = abs(df['close'] - df['open'])
        df['Candle_Range'] = df['high'] - df['low']
        df['Liquidity_Zone'] = (df['Candle_Body'] / df['Candle_Range']) > threshold
        return df
    
    def generate_fractal_signal(df):
        df['Signal'] = None
        for i in range(2, len(df) - 2):
            if df['High_Fractal'].iloc[i] and df['Liquidity_Zone'].iloc[i]:
                df.at[i, 'Signal'] = 'Sell'
            elif df['Low_Fractal'].iloc[i] and df['Liquidity_Zone'].iloc[i]:
                df.at[i, 'Signal'] = 'Buy'
        return df
    
    def calculate_stochastic_with_slowing(df, k_period=5, d_period=3, slowing=3):
        """
        Calculate the Stochastic Oscillator (%K and %D) with slowing for crossover detection.

        Parameters:
        - df: DataFrame containing 'high', 'low', and 'close' prices.
        - k_period: Lookback period for %K (default is 5).
        - d_period: Period for %D smoothing (default is 3).
        - slowing: Slowing period applied to %K (default is 3).

        Returns:
        - DataFrame with %K and %D columns.
        """
        # Ensure sufficient rows to calculate Stochastic Oscillator
        if len(df) < k_period + slowing + d_period:
            raise ValueError("DataFrame does not contain enough rows for calculation")

        # Calculate the highest high and lowest low over the last k_period periods
        df['low_k'] = df['low'].rolling(window=k_period, min_periods=1).min()
        df['high_k'] = df['high'].rolling(window=k_period, min_periods=1).max()

        # Calculate raw %K: (close - lowest low) / (highest high - lowest low) * 100
        df['%K_raw'] = 100 * ((df['close'] - df['low_k']) / (df['high_k'] - df['low_k']))

        # Apply slowing: simple moving average of %K over the slowing period
        df['%K'] = df['%K_raw'].rolling(window=slowing, min_periods=1).mean()

        # Calculate %D: moving average of %K over d_period
        df['%D'] = df['%K'].rolling(window=d_period, min_periods=1).mean()

        # Drop intermediate columns
        df.drop(columns=['low_k', 'high_k', '%K_raw'], inplace=True)

        return df[['%K', '%D']]
    
    def detect_stoch_crossover(df):
        df = df.copy()
        overbought = 80
        oversold = 20
        
        df.loc[:,'OverBought'] = (df['%K'] > overbought).astype(int)
        df.loc[:,'OverSold'] = (df['%K'] < oversold).astype(int)
        
        df.loc[:,'Buy_signal'] = ((df['OverSold'] == 1) & (df['%K'].shift(1) < df['%D'].shift(1)) & (df['%K'] > df['%D'])).astype(int)
        df.loc[:,'Sell_signal'] = ((df['OverBought'] == 1) & (df['%K'].shift(1) > df['%D'].shift(1)) & (df['%K'] < df['%D'])).astype(int)
        
        # df.loc[:,'Buy_signal'] = ((df['OverBought'] == 1) & (df['%K'].shift(1) < df['%D'].shift(1)) & (df['%K'] > df['%D'])).astype(int)
        # df.loc[:,'Sell_signal'] = ((df['OverSold'] == 1) & (df['%K'].shift(1) > df['%D'].shift(1)) & (df['%K'] < df['%D'])).astype(int)
        return df[['%K', '%D','Buy_signal','Sell_signal']]

    def calculate_sma(data, window):
        sma_values = np.convolve(data, np.ones(window), 'valid') / window
        # Pad the result with None to align with the original data length
        new_sma_values = np.concatenate((np.full(window-1, None), sma_values))
        # Handle NaN values
        # Create a DataFrame with the original data and SMA values
        df_sma = pd.DataFrame({'data': data, 'sma': new_sma_values})
        
        # Fill NaN values in 'sma' column using forward fill for the initial NaN values
        df_sma['sma'].fillna(method='ffill', inplace=True)
        # Fill NaN values in 'sma' column using backward fill for the remaining NaN values
        df_sma['sma'].fillna(method='bfill', inplace=True)
        
        return new_sma_values[-1]
    
    def bollinger(data,window=20,std_dev=2):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (std_dev * rolling_std)
        lower_band = rolling_mean - (std_dev * rolling_std)
        return upper_band,lower_band
    
    def calculate_rsi(data,period=14):
        """
        Calculate the Relative Strength Index (RSI) for a given dataset.

        Parameters:
        data (pd.Series or np.array): A pandas Series or numpy array containing the price data.
        period (int): The number of periods to use for the RSI calculation. Default is 14.

        Returns:
        pd.Series: A pandas Series containing the RSI values.
        """
        delta = data.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        # Initial average gain and loss
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)

        avg_gain[period] = np.mean(gain[1:period + 1])
        avg_loss[period] = np.mean(loss[1:period + 1])

        # Calculate the smoothed average gains and losses
        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
            
        avg_loss = np.where(avg_loss == 0, np.nan,avg_loss)
        

        # Avoid division by zero
        rs = np.divide(avg_gain,avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        rsi = np.where(np.isnan(rsi) & (avg_loss == 0),100,rsi)
        rsi = np.where(np.isnan(rsi) & (avg_gain == 0),100,rsi)

        # Truncate the RSI to start at the 'period' index
        rsi[:period] = np.nan

        return pd.Series(rsi, name='RSI')
    
    
    def get_RSI_SLOPE(data, period=14, slope_period=3):
        rsi = Indicators.calculate_rsi(data)
        rsi_slope = rsi.diff(slope_period) / slope_period
        
        return pd.Series(rsi_slope,name='RSI_slope')
        
    

    def get_dom(symbol):
        """
        Retrieve Depth of Market (DOM) data for a given symbol.

        Parameters:
        symbol (str): The symbol for which to retrieve DOM data (e.g., 'EURUSDm').

        Returns:
        pd.DataFrame: A DataFrame containing the bid and ask prices and volumes.
        """
        # # Ensure the symbol is selected
        # if not mt5.symbol_select(symbol, True):
        #     raise ValueError(f"Failed to select symbol {symbol}")

        # Get the DOM data
        dom = mt5.market_book_get(symbol)
        
        if dom is None:
            raise ValueError(f"Failed to retrieve DOM data for symbol {symbol}")

        # Create DataFrame from DOM data
        bids = []
        asks = []
        
        for entry in dom:
            if entry['type'] == mt5.ORDER_BUY:
                bids.append({
                    'price': entry['price'],
                    'volume': entry['volume']
                })
            elif entry['type'] == mt5.ORDER_SELL:
                asks.append({
                    'price': entry['price'],
                    'volume': entry['volume']
                })
        
        # Convert to DataFrame
        bids_df = pd.DataFrame(bids).sort_values(by='price', ascending=False).reset_index(drop=True)
        asks_df = pd.DataFrame(asks).sort_values(by='price', ascending=True).reset_index(drop=True)

        return bids_df, asks_df

    
class CandleSticks():
    @staticmethod
    def evening_star(candles):
        first_candle = candles[-3]
        second_candle = candles[-2]
        third_candle = candles[-1]
        if first_candle['close'] < first_candle['open'] and third_candle['close'] < first_candle['open'] and second_candle['close'] > first_candle['close'] and third_candle['close'] < second_candle['open'] and third_candle['close'] > second_candle['close']:
            return True
        return False
    
    # def morning_star(candles):
    #     first_candle = candles[-3]
    #     second_candle = candles[-2]
    #     third_candle = candles[-1]
    #     if first_candle['close'] > first_candle['open'] and third_candle['close'] > first_candle['open'] and second_candle['close'] < first_candle['close'] and third_candle['close'] > second_candle['open'] and third_candle['close'] < second_candle['close']:
    #         return True
    #     return False

    def morning_star(candles, bullish_threshold=0.5):
        """
        Check if the last three closed candles form a Morning Star pattern.

        Parameters:
            candles (list): List of dictionaries with candle data (must include 'open', 'close', 'high', 'low').
            bullish_threshold (float): The proportion of the first candle's body that the third candle should close above to confirm the pattern.

        Returns:
            bool: True if the last three closed candles form a Morning Star pattern, False otherwise.
        """
        if len(candles) < 4:
            raise ValueError("Not enough candle data to determine Morning Star pattern")
        
        # Use the last three *closed* candles
        first_candle = candles[-4]  # Large bearish candle
        second_candle = candles[-3]  # Small candle (Doji, Spinning Top, etc.)
        third_candle = candles[-2]  # Large bullish candle
        
        # First Candle: Large bearish body
        first_body = first_candle['open'] - first_candle['close']
        if first_body <= 0:
            return False  # Not a bearish candle
        
        # Second Candle: Small body (indecision candle)
        second_body = abs(second_candle['close'] - second_candle['open'])
        second_range = second_candle['high'] - second_candle['low']
        if second_body > 0.5 * second_range:
            return False  # Not a small indecision candle
        
        # Third Candle: Large bullish body
        third_body = third_candle['close'] - third_candle['open']
        if third_body <= 0:
            return False  # Not a bullish candle
        
        # Check if the third candle closes above the midpoint of the first candle's body
        midpoint_first_candle = (first_candle['open'] + first_candle['close']) / 2
        if third_candle['close'] <= midpoint_first_candle + bullish_threshold * abs(first_body):
            return False  # Third candle does not close sufficiently above the first candle's midpoint
        
        return True


    # def doji(candles,threshold=0.01):
    #     candle = candles[-2]
    #     body_len = abs(candle['close'] - candle['open']) 
    #     candle_range =  (candle['high'] - candle['low'])
    #     if body_len <= threshold * candle_range and candle_range > 0 :
    #         return True
    #     return False
    
    
    def doji(candles, threshold=0.01):
        """
        Check if the second-to-last candle in the list is a Doji candlestick.

        Parameters:
            candles (list): List of dictionaries with candle data (must include 'open', 'close', 'high', 'low').
            threshold (float): Proportion of the candle's range that determines if the body is sufficiently small to be considered a Doji.

        Returns:
            bool: True if the second-to-last candle is a Doji, False otherwise.
        """
        if len(candles) < 2:
            raise ValueError("Not enough candle data to determine Doji pattern")
        
        candle = candles[-2]  # Use the second-to-last candle
        
        # Calculate the body length and the total candle range
        body_len = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        # Ensure the range is positive to avoid division by zero
        if candle_range <= 0:
            return False
        
        # Check if the body is small relative to the range
        if body_len <= threshold * candle_range:
            return True
        
        return False

    
    def shooting_star(candles):
        candle = candles[-1]
        if candle['close'] < candle['open'] and (candle['close'] - candle['low']) >= 2 * (candle['open'] - candle['close']) and (candle['high'] - candle['open']) <= 0.1 * (candle['high'] - candle['low']):
            return True
        return False
    
    # def hammer(candles):
    #     candle = candles[-2]
    #     if candle['close'] > candle['open'] and (candle['high'] - candle['close']) >= 2 * (candle['open'] - candle['close']) and (candle['open'] - candle['low']) <= 0.1 * (candle['high'] - candle['low']):
    #         return True
    #     return False
    
    def is_hammer(candle, threshold=0.01):
        """Detect Hammer candlestick pattern."""
        candle = candle[-2]
        body_length = abs(candle['close'] - candle['open'])
        lower_shadow = candle['open'] - candle['low'] if candle['open'] > candle['close'] else candle['close'] - candle['low']
        upper_shadow = candle['high'] - candle['close'] if candle['open'] > candle['close'] else candle['high'] - candle['open']
        candle_range = candle['high'] - candle['low']
        
        # Check if the body is small relative to the total range
        if body_length <= threshold * candle_range and lower_shadow >= 2 * body_length and upper_shadow <= 0.1 * body_length:
            return True
        return False

        
    def bearish_engulfing(candles):
        if len(candles) < 2:
            return False

        previous_candle = candles[-3]
        current_candle = candles[-2]

        if (previous_candle['close'] > previous_candle['open'] and
            current_candle['close'] < previous_candle['open'] and
            current_candle['open'] > previous_candle['close'] and
            current_candle['close'] < previous_candle['open'] and
            current_candle['open'] > previous_candle['close']):
            return True

        return False

    def bullish_engulfing(candles):
        if len(candles) < 2:
            return False

        previous_candle = candles[-2]
        current_candle = candles[-1]

        if (previous_candle['close'] < previous_candle['open'] and
            current_candle['close'] > previous_candle['open'] and
            current_candle['open'] < previous_candle['close'] and
            current_candle['close'] > previous_candle['open'] and
            current_candle['open'] < previous_candle['close']):
            return True

        return False

    def dark_cloud_cover(candles):
        prev_candle = candles[-2]
        curr_candle = candles[-1]
        if(
            curr_candle['close'] < curr_candle['open'] and prev_candle['close'] > prev_candle['open'] and curr_candle['open'] > prev_candle['high'] and curr_candle['close'] < (prev_candle['open'] + prev_candle['close']) / 2
        ):
            return True

        else:
            return False

    # def morning_star(candles):
    #     prev_prev_candle = candles[-3]
    #     prev_candle = candles[-2]
    #     curr_candle = candles[-1]
    #     if(
    #         prev_candle['close'] < prev_candle['open'] and abs(curr_candle['open'] - curr_candle['close']) < 0.1 * curr_candle['close'] and curr_candle['close'] > prev_candle['open'] and curr_candle['open'] < prev_candle['close'] and prev_prev_candle['close'] > prev_prev_candle['open'] and prev_candle['open'] > prev_prev_candle['close']
    #     ):
    #         return True

    #     else:
    #         return False

    def bullish_harami(candle):
        prev_candle = candle[-2]
        if(
            candle['open'] > prev_candle['close'] and candle['close'] < prev_candle['open'] and abs(candle['open'] - candle['close']) < abs(prev_candle['open'] - prev_candle['close'])
        ):
            return True

        else:
            return False

    def bearish_harami(candle):
        prev_candle = candle[-2]
        if(
            candle['open'] < prev_candle['close'] and candle['close'] > prev_candle['open'] and abs(candle['open'] - candle['close']) < abs(prev_candle['open'] - prev_candle['close'])
        ):
            return True

        else:
            return False

    def bullish_marubozu(candle):
        if(
            candle['close'] > candle['open'] and abs(candle['close'] - candle['open']) < 0.05 * candle['close'] and candle['high'] == candle['close'] and candle['low'] == candle['open']
        ):
            return True

        else:
            return False

    def bearish_marubozu(candle):
        if candle['close'] < candle['open'] and abs(candle['close'] - candle['open']) < 0.05 * candle['close'] and candle['low'] == candle['close'] and candle['high'] == candle['open']:
            return True
        else:
            return False

    def rising_three_methods(candles):
        if len(candles) < 5:
            return False
        
        first_candle = candles[-5]
        last_candle = candles[-1]
        
        if first_candle['close'] < first_candle['open'] or last_candle['close'] < last_candle['open']:
            return False
        
        for i in range(-4, 0):
            if candles[i]['close'] > first_candle['close'] or candles[i]['close'] > candles[i-1]['close']:
                return False
        
        return last_candle['low'] > first_candle['low']

    def three_white_soldiers(candles):
        if len(candles) < 3:
            return False
        
        for i in range(2, len(candles)):
            if candles[i]['close'] <= candles[i-1]['close'] or candles[i-1]['close'] <= candles[i-2]['close']:
                return False
        
        return True

    def three_black_crows(candles):
        if len(candles) < 3:
            print('Insufficient data for Three Black Crows pattern')
            return False
        
        recent_candles = candles[-3:]

        for i in range(2, len(recent_candles)):
            # Check if each candle is bearish
            if recent_candles[i]['close'] >= recent_candles[i - 1]['close'] or recent_candles[i - 1]['close'] >= recent_candles[i - 2]['close']:
                # print('No Three Black Crows pattern detected')
                return False

        # print('Three Black Crows pattern detected')
        return True

    def calculate_kijun_sen(df, period=26):
        # Calculate the Kijun-sen line
        df['Kijun_sen'] = (df['high'].rolling(window=period).max() + df['low'].rolling(window=period).min()) / 2
        # df['Kijun-sen'].bfill(inplace=True)
        return df

    def detect_crossover(df):
        # Detect crossover between the current price and the Kijun-sen line
        df['kijun_sen_signal'] = np.where((df['close'] > df['Kijun_sen']) & (df['close'].shift(1) <= df['Kijun_sen'].shift(1)), 'Bullish_Crossover', 
                            np.where((df['close'] < df['Kijun_sen']) & (df['close'].shift(1) >= df['Kijun_sen'].shift(1)), 'Bearish_Crossover', 'No_Signal'))
        return df

    def bullish_belt_hold(candle):
        if(
            candle['close'] > candle['open'] and candle['open'] == candle['low'] and (candle['close'] - candle['open']) > 2 * (candle['high'] - candle['close'])
        ):
            print('buy')

        else:
            print('No signal')

    def tweezer_top(candle, prev_candle):
        if(
            candle['high'] == prev_candle['high'] and candle['low'] > prev_candle['low']
        ):
            print('buy')

        else:
            print('No signal')

    def bullish_tri_star(candles):
        if len(candles) != 3:
            return False
        
        first_candle = candles[0]
        second_candle = candles[1]
        third_candle = candles[2]
        
        if(
            first_candle['close'] < first_candle['open'] and second_candle['close'] < second_candle['open'] and third_candle['close'] > third_candle['open'] and first_candle['high'] > second_candle['high'] and second_candle['high'] < third_candle['low'] and first_candle['low'] < second_candle['low'] and second_candle['low'] > third_candle['high']
        ):
            print('buy')

        else:
            print('No signal')

#     def engulfing_pattern(df):
        
#         # Define the style of the candlestick plot
#         mc = mpf.make_marketcolors(up='g', down='r')
#         s = mpf.make_mpf_style(marketcolors=mc)
#         # Detect bullish engulfing pattern
#         engulfing_pattern = mpf.make_addplot((df['open'] > df['close'].shift()) & (df['close'] > df['open'].shift()), type='scatter', markersize=100, marker='^', color='green')
#         mpf.plot(df, type='candle', style=s,addplot=engulfing_pattern)



    