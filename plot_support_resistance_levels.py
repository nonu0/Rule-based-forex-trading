import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
from initialiser import MetaTrader5Client
from indicators import CurrentRates

account_number = 179196033
# account_number = 180299878
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'

client = MetaTrader5Client(account_number,account_password,account_server)
client.login()

# symbols = [
#            'EURUSDm','EURAUDm','USDCADm', 
#            'GBPUSDm','AUDUSDm','USDCADm','NZDUSDm',
#            'USDCHFm','EURGBPm',]

symbol = 'EURGBPm'

df = CurrentRates.get_daily_rates(symbol)
# print(df)
df = df[df['tick_volume'] != 0].reset_index(drop=True)

# Parameters
n1, n2 = 2, 2
start_idx, end_idx = 3, 205
min_distance = 0.0009  # Minimum distance between levels
touch_threshold = 6  # Minimum number of touches to consider a level significant

# Support and resistance level functions
def find_levels_optimized(df, start_idx, end_idx, n1, n2):
    """Find support and resistance levels."""
    lows = df['low'].values
    highs = df['high'].values
    support_levels = []
    resistance_levels = []

    for i in range(start_idx, end_idx):
        if all(lows[i] <= lows[i - j] for j in range(1, n1 + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, n2 + 1)):
            support_levels.append((i, lows[i]))

        if all(highs[i] >= highs[i - j] for j in range(1, n1 + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, n2 + 1)):
            resistance_levels.append((i, highs[i]))

    return support_levels, resistance_levels

def reduce_close_levels(levels, min_distance):
    """Remove levels too close to each other."""
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: x[1])
    reduced_levels = [levels[0]]
    for i in range(1, len(levels)):
        if abs(levels[i][1] - reduced_levels[-1][1]) >= min_distance:
            reduced_levels.append(levels[i])
    return reduced_levels

def significance_score(level, df, min_distance, touch_threshold, window=50):
    """Calculate significance score based on frequency of price touches."""
    touch_count = 0
    for i in range(level[0] - window, level[0] + window):
        if i < 0 or i >= len(df):
            continue
        if abs(df['low'][i] - level[1]) < min_distance or abs(df['high'][i] - level[1]) < min_distance:
            touch_count += 1
    # Only return level if it meets the touch threshold
    return touch_count >= touch_threshold

# Find and reduce levels
support_levels, resistance_levels = find_levels_optimized(df, start_idx, end_idx, n1, n2)
support_levels = reduce_close_levels(support_levels, min_distance)
resistance_levels = reduce_close_levels(resistance_levels, min_distance)

# Filter support and resistance levels by significance (touch threshold)
support_levels = [level for level in support_levels if significance_score(level, df, min_distance, touch_threshold)]
resistance_levels = [level for level in resistance_levels if significance_score(level, df, min_distance, touch_threshold)]

# Enhanced plotting function
def plot_candlestick_with_annotations(df, support_levels, resistance_levels, start, end):
    """Plot candlestick chart with enhanced features."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index[start:end],
        open=df['open'][start:end],
        high=df['high'][start:end],
        low=df['low'][start:end],
        close=df['close'][start:end],
    )])

    # Add support levels with annotations
    for level in support_levels:
        fig.add_shape(
            type='line',
            x0=level[0], y0=level[1],
            x1=end, y1=level[1],
            line=dict(color="red", width=2, dash="dash")
        )
        fig.add_annotation(
            x=level[0], y=level[1],
            text=f"Support: {level[1]:.5f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="red",
            font=dict(size=10, color="red"),
            ax=20, ay=0
        )

    # Add resistance levels with annotations
    for level in resistance_levels:
        fig.add_shape(
            type='line',
            x0=level[0], y0=level[1],
            x1=end, y1=level[1],
            line=dict(color="green", width=2, dash="dash")
        )
        fig.add_annotation(
            x=level[0], y=level[1],
            text=f"Resistance: {level[1]:.5f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="green",
            font=dict(size=10, color="green"),
            ax=20, ay=0
        )

    # Add titles and axis labels
    fig.update_layout(
        title="Candlestick Chart with Support and Resistance Levels",
        xaxis_title="Index",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    # Enable zoom and pan
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_yaxes(fixedrange=False)

    fig.show()

# Plot the enhanced chart with prioritized levels
# plot_candlestick_with_annotations(df, support_levels, resistance_levels, 0, 200)