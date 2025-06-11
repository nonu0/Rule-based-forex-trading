import pandas as pd
import MetaTrader5 as mt5
import logging
import time
from indicators import CurrentRates, CandleSticks, Indicators
from initialiser import MetaTrader5Client
from helper_functions import (
    place_buy_order, update_trade_result, calculate_buy_stop_loss_take_profit,
    calculate_atr, calculate_sell_stop_loss_take_profit, place_sell_order,
    log_trade, modify_stop_loss, initialise_log_file
)

account_number = 179196033
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'

client = MetaTrader5Client(account_number, account_password, account_server)
client.login()

symbols = [
    'EURUSDm', 'EURJPYm', 'EURAUDm', 'USDJPYm', 
    'GBPUSDm', 'AUDUSDm', 'NZDUSDm', 'USDCHFm', 
    'EURGBPm', 'BTCUSDm'
]

log_filename = r'C:\Users\Administrator\work\forexBot\core\logs\forex_trades.log'
initialise_log_file(log_filename)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

try:
    while True:
        for symbol in symbols:
            lot_size = 0.01
            tick_info = mt5.symbol_info_tick(symbol)
            if tick_info is None:
                logging.error(f"{symbol} symbol_info_tick is None. Skipping...")
                continue
            bid_price = tick_info.bid
            ask_price = tick_info.ask

            df = CurrentRates.get_15min_rates(symbol)
            # fractals = Indicators.get_fractals(df)
            df['highest'] = df['high'].rolling(window=5, center=True).max()
            df['lowest'] = df['low'].rolling(window=5, center=True).min()
            # print(df)
            df['High_Fractal'] = df['high'] == df['highest']
            df['Low_Fractal'] = df['low'] == df['lowest']
            recent_frac = df.iloc[-1]
            # print(recent_frac)
            atr = calculate_atr(df)
            buy_sl, buy_tp = calculate_buy_stop_loss_take_profit(bid_price, atr)
            sell_sl, sell_tp = calculate_sell_stop_loss_take_profit(ask_price, atr)

            try:
                if recent_frac['High_Fractal']:
                    print(f'{symbol} High Fractal detected')
                    result = place_sell_order(lot_size, ask_price, symbol, sell_sl, sell_tp)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"{symbol}: SELL order placed with SL: {sell_sl}, TP: {sell_tp}")
                        log_trade(symbol, result.order, 'SELL', 'High fractal detected', sell_sl, sell_tp, 'TBD', log_filename)
                    else:
                        logging.error(f"{symbol}: SELL order failed with error code {result.retcode if result else 'None'}")
                elif recent_frac['Low_Fractal']:
                    print(f'{symbol} Low Fractal detected')
                    result = place_buy_order(lot_size, bid_price, symbol, buy_sl, buy_tp)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        logging.info(f"{symbol}: BUY order placed with SL: {buy_sl}, TP: {buy_tp}")
                        log_trade(symbol, result.order, 'BUY', f"Low fractal detected at {recent_frac['close']}", buy_sl, buy_tp, 'TBD', log_filename)
                    else:
                        logging.error(f"{symbol}: BUY order failed with error code {result.retcode if result else 'None'}")
                else:
                    print(f'{symbol}: No Fractal detected')
            except Exception as e:
                logging.error(f"{symbol}: Error occurred while placing order: {e}")
        time.sleep(5)
except KeyboardInterrupt:
    logging.info("Program stopped manually.")
except Exception as e:
    logging.error(f"Critical error: {e}")
