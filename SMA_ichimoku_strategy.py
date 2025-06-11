import MetaTrader5 as mt5

import time

from initialiser import MetaTrader5Client
from indicators import CandleSticks, CurrentRates, Indicators


account_number = 179196033
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'
symbols = [
    'EURUSDm','EURJPYm','EURAUDm','USDCADm', 
           'USDJPYm','GBPUSDm','AUDUSDm','USDCADm',
           'USDCHFm','EURGBPm','USDKESm'
]
lots = 0.05
deviation = 20


client = MetaTrader5Client(account=account_number,password=account_password,server=account_server)
client.login()


cycle_count = 0
last_order_time = {symbol:0 for symbol in symbols}

five_min = CurrentRates.get_5min_rates('EURJPYm')
ichimoku = Indicators.ichimoku_strategy(five_min)
ichimoku = ichimoku.drop(columns=['Above_Cloud','Below_Cloud'])

def ichimoku_s(df):
    """
    Implement Ichimoku strategy with cloud position, span checks, and retracement for buy/sell signals.
    
    Parameters:
        df (DataFrame): DataFrame containing OHLC and Ichimoku indicator values.
        
    Returns:
        DataFrame: DataFrame with buy/sell signals added.
    """
    df['Signal'] = 0  # Initialize signal column, 0 means no signal
    
    for i in range(len(df)):
        # Check if current price is above or below the cloud
        if df['close'].iloc[i] > df['SpanA'].iloc[i] and df['close'].iloc[i] > df['SpanB'].iloc[i]:
            # Buy conditions: Price above cloud and Span A > Span B
            if df['SpanA'].iloc[i] > df['SpanB'].iloc[i]:
                # Check if there's a recent crossing from below the cloud and retrace to Span A
                if df['close'].iloc[i-1] < df['SpanA'].iloc[i-1] and df['close'].iloc[i] > df['SpanA'].iloc[i]:
                    # Retracement condition
                    retrace_index = i - 1
                    while retrace_index > 0 and df['close'].iloc[retrace_index] < df['SpanA'].iloc[retrace_index]:
                        retrace_index -= 1
                    if retrace_index != i - 1 and df['close'].iloc[retrace_index] > df['SpanA'].iloc[retrace_index]:
                        # df['Signal'].iloc[i] = 1  # Buy signal
                        df.loc[i,'Signal'] = 1

        elif df['close'].iloc[i] < df['SpanA'].iloc[i] and df['close'].iloc[i] < df['SpanB'].iloc[i]:
            # Sell conditions: Price below cloud and Span B > Span A
            if df['SpanB'].iloc[i] > df['SpanA'].iloc[i]:
                # Check if there's a recent crossing from above the cloud and retrace to Span A
                if df['close'].iloc[i-1] > df['SpanA'].iloc[i-1] and df['close'].iloc[i] < df['SpanA'].iloc[i]:
                    # Retracement condition
                    retrace_index = i - 1
                    while retrace_index > 0 and df['close'].iloc[retrace_index] > df['SpanA'].iloc[retrace_index]:
                        retrace_index -= 1
                    if retrace_index != i - 1 and df['close'].iloc[retrace_index] < df['SpanA'].iloc[retrace_index]:
                        # df['Signal'].iloc[i] = -1  # Sell signal
                        df.loc[i,'Signal']
    
    return df

# Usage example
# Assuming `df` contains the OHLC data and calculated Ichimoku indicators (SpanA, SpanB, etc.)
df = ichimoku_s(ichimoku)
print(df[['time','close', 'SpanA', 'SpanB', 'Signal']].tail(50))
