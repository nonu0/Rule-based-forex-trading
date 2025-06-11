from initialiser import MetaTrader5Client
from indicators import CurrentRates,Indicators

import plotly.graph_objects as go

# MetaTrader 5 Login
account_number = 179196033
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'

client = MetaTrader5Client(account=account_number,password=account_password,server=account_server)

client.login()
symbol = 'EURJPYm'
df = CurrentRates.get_15min_rates(symbol) 
fractals = Indicators.get_fractals(df)
liquidity = Indicators.liquidity_zones(fractals)
signal = Indicators.generate_fractal_signal(liquidity)
print(fractals.tail(50))

high_fractals = df[df['High_Fractal']]
low_fractal = df[df['Low_Fractal']]
# print(low_fractal)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x = df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close = df['close'],
    name='Candlesticks'
))

for idx, row in high_fractals.iterrows():
    fig.add_annotation(
        x=row['time'],
        y=row['high'],
        text=f"High {row['high']:.5f}",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowcolor='green',
        ax=0,
        ay=-20,
    )
    
for idx, row in low_fractal.iterrows():
    fig.add_annotation(
        x=row['time'],
        y=row['low'],
        text=f"Low {row['low']:.5f}",
        showarrow=True,
        arrowhead=3,
        arrowsize=2,
        arrowcolor='red',
        ax=0,
        ay=20,
    )

fig.update_layout(
    title='Fractals visualization',
    xaxis_title='Time',
    yaxis_title='Price',
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
)

fig.show()