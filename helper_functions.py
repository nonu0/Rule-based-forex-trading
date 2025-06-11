from datetime import datetime
import logging
import re
import os

import MetaTrader5 as mt5
import numpy as np

from indicators import Indicators,CurrentRates

def EMA(df):
    # for span in spans:
    df[f'EMA_20'] = df['close'].ewm(span=20,adjust=False).mean()
    df[f'EMA_50'] = df['close'].ewm(span=50,adjust=False).mean()
    df[f'EMA_200'] = df['close'].ewm(span=200,adjust=False).mean()
    # print(df)
    return df





def vwap_calculation(price,df):
    volumes = df['tick_volume']
    cummulative_volume = volumes.cumsum()
    cummulative_volume_prices = (price * volumes).cumsum()
    vwap = cummulative_volume_prices / cummulative_volume
    
    return vwap

def predict_trend(df):
    conditions =  [
         (df['close'] > df['EMA_20']) & (df['close'] > df['EMA_50']) & (df['close'] > df['EMA_200']) & (df['EMA_20'] > df['EMA_50']) & (df['EMA_50'] > df['EMA_200']),
        (df['close'] < df['EMA_20']) & (df['close'] < df['EMA_50']) & (df['close'] < df['EMA_200']) & (df['EMA_20'] < df['EMA_50']) & (df['EMA_50'] < df['EMA_200']),
    
    ]
    choices = ['bullish','bearish']
    df['Trend'] = np.select(conditions,choices,default='Neutral')
    trend = np.select(conditions,choices,default='Neutral')
    # print(df[100:])
    # print('trend',trend)
    return trend


def predict_hourly_trend(df):
    conditions =  [
         (df['close'] > df['EMA_200']),
         (df['close'] < df['EMA_200']) ,
    
    ]
    choices = ['bullish','bearish']
    df['Trend'] = np.select(conditions,choices,default='Neutral')
    trend = np.select(conditions,choices,default='Neutral')
    # print(df[100:])
    # print('trend',trend)
    return trend

def predict_15min_trend(df):
    conditions =  [
         (df['close'] > df['EMA_51']),
         (df['close'] < df['EMA_51']) ,
    
    ]
    choices = ['bullish','bearish']
    df['Trend'] = np.select(conditions,choices,default='Neutral')
    trend = np.select(conditions,choices,default='Neutral')
    # print(df[100:])
    # print('trend',trend)
    return trend

def is_level(df, l, n1, n2, level_type='support'):
    """
    Check if a level is support or resistance.

    Args:
        df (DataFrame): The data containing OHLC.
        l (int): The index of the candle to evaluate.
        n1 (int): Number of candles before the target candle.
        n2 (int): Number of candles after the target candle.
        level_type (str): 'support' or 'resistance'.

    Returns:
        bool: True if the level qualifies, otherwise False.
    """
    if level_type == 'support':
        for i in range(l - n1 + 1, l + 1):
            if df.low[i] > df.low[i - 1]:
                return False
        for i in range(l + 1, l + n2 + 1):
            if df.low[i] < df.low[i - 1]:
                return False
    elif level_type == 'resistance':
        for i in range(l - n1 + 1, l + 1):
            if df.high[i] < df.high[i - 1]:
                return False
        for i in range(l + 1, l + n2 + 1):
            if df.high[i] > df.high[i - 1]:
                return False
    return True


def count_touches(df, level, tolerance=0.001):
    """
    Count the number of touches for a level within a given tolerance.

    Args:
        df (DataFrame): The visible data range (dfpl).
        level (float): The level price.
        tolerance (float): The acceptable price deviation for a touch.

    Returns:
        int: Number of touches within the visible range.
    """
    return ((df['low'] >= level - tolerance) & (df['low'] <= level + tolerance)).sum() + \
           ((df['high'] >= level - tolerance) & (df['high'] <= level + tolerance)).sum()
           
           
# Merge close levels
def merge_close_levels(levels, tolerance=0.001):
    merged_levels = []
    levels = sorted(levels, key=lambda x: x[2])  # Sort by price

    for level in levels:
        if not merged_levels or (level[0] == merged_levels[-1][0] and abs(level[2] - merged_levels[-1][2]) > tolerance):
            merged_levels.append(level)
        elif level[0] == merged_levels[-1][0]:  # Merge only same types
            if level[3] > merged_levels[-1][3]:
                merged_levels[-1] = level
    return merged_levels

def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) for a given symbol and timeframe.

    Parameters:
    - symbol: The symbol to calculate ATR for.
    - timeframe: The timeframe to use for ATR calculation (default is 'D1' - daily).
    - period: The period over which to calculate ATR (default is 14).

    Returns:
    - The ATR value.
    """
    df['TR1'] = df['high'] - df['low']
    df['TR2'] = abs(df['high'] - df['close'].shift(1))
    df['TR3'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['TR1','TR2','TR3']].max(axis=1)

    atr = (df['TR'].rolling(window=period).mean())
    df['ATR'] = (df['TR'].rolling(window=period).mean())
    # df.dropna(inplace=True)
    # print(df)
    # print('atr',atr)
    return atr

# def atr_midpoint(df,period=14):

def get_RSI(df,period=14):
    delta = df['close'].diff()
    # separate gains and losses 
    gain = (delta.where(delta > 0,0)).fillna(0)
    loss = (-delta.where(delta < 0,0)).fillna(0)
    
    # calculate the average gain and loss
    avg_gain = gain.rolling(window=period,min_periods=1).mean()
    avg_loss = loss.rolling(window=period,min_periods=1).mean()
    
    # calcultae rsi 
    rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    
    return df

def get_RSI_slope(df,period=14,slope_period=3):
    df['RSI'] = get_RSI(df,period=period)['RSI']
    df['RSI_SLOPE'] = df['RSI'].diff(periods=slope_period)
    return df

def calculate_buy_stop_loss_take_profit(price, atr, stop_loss_multiple=1, take_profit_multiple=1.5):
    """
    Calculate stop loss and take profit levels based on ATR.

    Parameters:
    - current_price: The current price of the asset.
    - atr: The Average True Range (ATR) value.
    - stop_loss_multiple: The multiple of ATR to use for setting stop loss (default is 2).
    - take_profit_multiple: The multiple of ATR to use for setting take profit (default is 3).

    Returns:
    - The stop loss and take profit levels.
    """
    atr_loss = atr * stop_loss_multiple
    atr_profit = atr * take_profit_multiple
    stop_loss = price - atr_loss
    take_profit = price + atr_profit
    sl = stop_loss.iloc[-1]
    tp = take_profit.iloc[-1]
    return sl, tp

def calculate_support_resistance(symbol, n1=10, n2=10):

    # Get daily rates for the specified symbol
    df = CurrentRates.get_daily_rates(symbol)

    # Define the pivot identification function
    def pivotid(df1, l, n1, n2):
        if l - n1 < 0 or l + n2 >= len(df1):
            return 0

        pividlow = 1
        pividhigh = 1
        for i in range(l - n1, l + n2 + 1):
            if df1.low[l] > df1.low[i]:
                pividlow = 0
            if df1.high[l] < df1.high[i]:
                pividhigh = 0
        if pividlow and pividhigh:
            return 3
        elif pividlow:
            return 1
        elif pividhigh:
            return 2
        else:
            return 0

    # Apply the pivot identification function
    df['pivot'] = df.apply(lambda x: pivotid(df, x.name, n1, n2), axis=1)

    # Get the pivot points
    def pointpos(x):
        if x['pivot'] == 1:
            return x['low'] - 1e-3  # Support level
        elif x['pivot'] == 2:
            return x['high'] + 1e-3  # Resistance level
        else:
            return np.nan

    df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
    
    return df

def calculate_sell_stop_loss_take_profit(price,atr,stop_loss_multiplier=1.5,take_profit_multiplier=3):
    atr_loss = atr * stop_loss_multiplier
    atr_profit = atr * take_profit_multiplier
    stop_loss = price + atr_loss
    take_profit = price - atr_profit
    sl = stop_loss.iloc[-1]
    tp = take_profit.iloc[-1]
    return sl,tp



# Function to place a buy order
def place_buy_order(lot, price,symbol,sl,tp, deviation=20):
# def place_buy_order(lot, price,symbol,point, deviation=20):
    # sl,tp = validate_sl_tp(symbol,price,sl,tp)
    buy_request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": sl,
    # "sl": price - 100 * point,
    "tp": tp,
    # "tp": price + 100 * point,
    "deviation": deviation,
    "magic": 234000,
    "comment": "python script open",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}
    result = mt5.order_send(buy_request)
    print("Buy order request:", buy_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode =", result.retcode)
        print("Failed order details:", result)
        # if trading is disabled 
        if result.retcode == mt5.TRADE_RETCODE_REJECT:
            print("\nTrading on this symbol is disabled",symbol)
            return
        else:
            print("Failed order details:",result)
            mt5.shutdown()
            quit()
    print("Buy order placed successfully:", result)
    
    return result

 
# Function to place a sell order
def place_sell_order(lot,price,symbol,sl,tp,deviation=20):
# def place_sell_order(lot,price,symbol,point,deviation):
    # sl,tp = validate_sl_tp(symbol,price,sl,tp)
    sell_request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_SELL,
    "price": price,
    "sl": sl,  # Stop Loss
    # "sl": price + 100 * point,  # Stop Loss
    # "tp": price - 100 * point,  # Take Profit
    "tp": tp,  # Take Profit
    "deviation": deviation,
    "magic": 234000,
    "comment": "python script close",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}
    result = mt5.order_send(sell_request)
    print("Sell order request:", sell_request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("Order send failed, retcode =", result.retcode)
        print("Failed order details:", result)
        if result.retcode == mt5.TRADE_RETCODE_REJECT:
            print("Trading is disabled ofr this symbol",symbol)
            return
        else:
            print("Fsiled trade details",result)
            mt5.shutdown()
            quit()
    print("Sell order placed successfully:", result)
    
    return result

def get_position():
    positions = mt5.positions_get()
    if not positions:
        print("No positions found")
        mt5.shutdown()
        quit()

    
def get_candle(data, index):
    """
    Get the candlestick at the specified index from the DataFrame.

    Parameters:
    - data: DataFrame containing historical candlestick data.
    - index: Index of the candlestick to retrieve.

    Returns:
    - A dictionary representing the candlestick at the specified index.
    """
    candle = {}
    if index >= 0 and index < len(data):
        candle['open'] = data.at[index, 'open']
        candle['high'] = data.at[index, 'high']
        candle['low'] = data.at[index, 'low']
        candle['close'] = data.at[index, 'close']
    return candle

def get_candles(data, start_index, end_index):
    """
    Get candlesticks within the specified range of indices from the DataFrame.

    Parameters:
    - data: DataFrame containing historical candlestick data.
    - start_index: Start index of the range.
    - end_index: End index of the range.

    Returns:
    - A list of dictionaries representing the candlesticks within the specified range.
    """
    candles = []
    for index in range(start_index, end_index + 1):
        candle = get_candle(data, index)
        candles.append(candle)
    return candles

def stoch_result(latest_k,latest_d):
    # Example trading strategy:
    if latest_k > 80 and latest_d > 80:
        print("Overbought condition: Consider selling or taking profits")
    elif latest_k < 20 and latest_d < 20:
        print("Oversold condition: Consider buying or looking for long opportunities")
    elif latest_k > latest_d:
        print("Bullish momentum: Consider buying or holding positions")
    elif latest_k < latest_d:
        print("Bearish momentum: Consider selling or shorting positions")
    else:
        print("No clear signal")

def log_trade(symbol,order_id,trade_type,reason,sl,tp,result,log_filename):
    try:
        with open(log_filename, 'a') as file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file.write(f"TimeStamp: {timestamp} Symbol: {symbol} Order ID: {order_id} Position: {trade_type} Reason: {reason} Stop Loss: {sl} take Profit: {tp} Result: {result}")
    except IOError as e:
        print(f'Error logging trade: {e}')
        
def update_trade_result(log_filename,order_id,result):
    pattern = re.compile(rf'Order ID: {order_id}, .*?Result:TBD')
    try:
        with open(log_filename,'r') as file:
            lines = file.readlines()
            
        with open(log_filename, 'w') as file:
            for line in lines:
                if pattern.search(line):
                    updated_line = re.sub(r"Result:TBD", f"Result:{result}",line)
                    file.write(updated_line)
                else:
                    file.write(line)
                    
    except IOError as e:
        print(f"Error updating trade result: {e}")
        
def initialise_log_file(log_filename):
    try:
        file_exists = os.path.isfile(log_filename)
        with open(log_filename, 'a') as file:
            if not file_exists:
                file.write('TimeStamp, Symbol, Order ID, Position, Reason, SL, TP, Result') # add headers if file empty
                
    except IOError as e:
        print(f"Error initializing log file: {e}")

def trailing_stop(positions,symbols,trailing_stop_value):
    for pos in positions:
        if pos.symbol in symbols:
            symbol = pos.symbol
            ticket = pos.ticket
            old_loss = pos.sl
            points = mt5.symbol_info(symbol).point
            trail = trailing_stop_value * points
            print('trail',trail)
            new_sl = mt5.symbol_info_tick(symbol).bid + trail
        return new_sl

def modify_stop_loss(symbol,order_ticket,new_sl,deviation=20):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol":symbol,
        "position":order_ticket,
        "sl":new_sl,
        "deviation":deviation,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Error: unable to modify stop loss for {symbol} with ticket {order_ticket}")
        
def calculate_avg_volume(symbol, period=20, timeframe=mt5.TIMEFRAME_M15):
    rates = mt5.copy_rates_from_pos(symbol,timeframe,0,period)
    if rates is not None and len(rates) == period:
        volume_sum = sum(rates['tick_volume'] for rate in rates)
        
        return volume_sum / period
    
def get_current_volume(symbol,timeframe=mt5.TIMEFRAME_M15):
    rates = mt5.copy_rates_from_pos(symbol,timeframe,0,1)
    if rates is not None and len(rates) > 0:
        return rates['tick_volume'].iloc[-1]
    else:
        print(f"Failed to retrieve current volume for {symbol}")
        return None
    
def close_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        logging.info(f'No open positions for {symbol}.')
        return False
    
    for position in positions:
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_BUY
        symbol = position.symbol
        volume = position.volume
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': volume,
            'type': order_type,
            'position': position.ticket,
            'price': mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
            'deviation': 10,  # Maximum price deviation in points
            'magic': 234000,  # A unique identifier for your trades, can be any number
            'comment': 'Early exit based on reversal indicators',
        }
        
        result  = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f'Successfully closed position for {symbol} - Ticket:{position.ticket}')
            return True
        else:
            logging.error(f'Failed to close position for {symbol} - Ticket: {position.ticket}')
            return False
    return False
        
def check_early_exit(symbol,rates,position_type,position_entry_price):
    VOLUME_THRESHOLD = 0.8
    
    volume = rates['tick_volume']
    current_volume = volume.iloc[-1]
    avg_volume = volume.rolling(window=20).iloc[-1]
    volume_check = avg_volume * VOLUME_THRESHOLD
    
    if position_type  == 'BUY':
        if current_volume < volume_check:
            print(f'{symbol} - Exiting BUY position early due to low volume.')
            close_position(symbol)
            
    elif position_type == 'SELL':
        if current_volume < volume_check:
            print(f'{symbol} - Exiting SELL position due to low volume')
            close_position(symbol)
            
            