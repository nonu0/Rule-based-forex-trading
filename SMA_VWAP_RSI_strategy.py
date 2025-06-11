import pandas as pd
import MetaTrader5 as mt5

import logging
import time

from indicators import CurrentRates,CandleSticks,Indicators
from initialiser import MetaTrader5Client
from helper_functions import (vwap_calculation,calculate_atr,
                              calculate_buy_stop_loss_take_profit,
                              calculate_sell_stop_loss_take_profit,
                              get_candles,EMA,predict_trend,place_buy_order,update_trade_result,
                              place_sell_order,log_trade,modify_stop_loss,initialise_log_file)


account_number = 179196033
# account_number = 180299878
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'

client = MetaTrader5Client(account_number,account_password,account_server)
client.login()


symbols = [
           'EURUSDm','EURJPYm','EURAUDm','USDCADm', 
           'USDJPYm','GBPUSDm','AUDUSDm','USDCADm','NZDUSDm',
           'USDCHFm','EURGBPm','BTCUSDm']

order_interval = 10800
trailing_stop_value = 0.11
risk_percent = 1
# lot_size = 0.01
log_filename = r'C:\Users\Administrator\work\forexBot\core\logs\forex_trades.log'
initialise_log_file(log_filename)

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler(log_filename),logging.StreamHandler()])

cycle_count = 0 
last_order_time = {symbol:0 for symbol in symbols}

try:
    while True:
        cycle_count += 1
        for symbol in symbols:
            current_time = time.time()
            if current_time - last_order_time[symbol] < order_interval:
                print(f'Skipping {symbol} for now. Order placed less than 3 hours ago.')
                continue
            
            print(f'\n======================Cycle {cycle_count}=======================')
            print(f'Looking for data on {symbol}')
            
            symbol_info = mt5.symbol_info(symbol)
            # print(symbol_info)
            if symbol_info is None:
                print(f'{symbol} is not visible, cannot call order check.')
                continue
            
            if not symbol_info.trade_mode:
                print(f'Trading not allowed for {symbol}. SKipping to next symbol.')
                continue
            
            if not symbol_info.visible:
                print(f'{symbol} is not visible, trying to turn on.')
                if not mt5.symbol_select(symbol,True):
                    print(f'Failed to turn on {symbol}.')
                    mt5.shutdown()
                    quit()
                    
            lot_size = 0.01
            # df_daily = CurrentRates.get_daily_rates(symbol)
            df_four_hour = CurrentRates.get_4hour_rates(symbol)
            df_one_hour = CurrentRates.get_hourly_rates(symbol)
            df_15min = CurrentRates.get_15min_rates(symbol)
            df_five_min = CurrentRates.get_5min_rates(symbol)
            # df_min = CurrentRates.get_minute_rates(symbol)
            
            bid_price = mt5.symbol_info_tick(symbol).bid
            ask_price = mt5.symbol_info_tick(symbol).ask
            # print('bid_price',bid_price)
            # print('ask_price',ask_price)
            
            vwap_sell = vwap_calculation(ask_price,df_one_hour).iloc[-1]
            vwap_buy = vwap_calculation(bid_price,df_one_hour).iloc[-1]
            
            stochastic = Indicators.calculate_stochastic_with_slowing(df_15min)
            # print(stochastic)
            
            stoch_crossover = Indicators.detect_stoch_crossover(stochastic)
            # print('stoch_crossover',stoch_crossover)
            
            atr = calculate_atr(df_one_hour)
            sell_sl,sell_tp = calculate_sell_stop_loss_take_profit(ask_price,atr)
            buy_sl,buy_tp = calculate_buy_stop_loss_take_profit(bid_price,atr)
            four_hour_ema = EMA(df_four_hour)
            hourly_trend = predict_trend(four_hour_ema)
            current_trend = hourly_trend[-1]
            # print(hourly_ema)
            # print(hourly_trend[-1])
            rsi = Indicators.calculate_rsi(df_one_hour['close'])
            latest_rsi = rsi.iloc[-1]
            print('latest_rsi',latest_rsi)
            
            try:
                if current_trend == 'bullish':
                    print(f'{symbol} is trending bullish.')
                    if (bid_price > vwap_buy) and (latest_rsi > 60):
                        print(f'{symbol} volume and RSI is bullish.')   
                        if (df_15min['%K'].iloc[-1] > 70) and (df_15min['%D'].iloc[-1] > 70) and stoch_crossover['Buy_signal'].iloc[-1] == 1:
                            print(f'{symbol} stochastic crossover bullish.')
                            result = place_buy_order(lot_size,bid_price,symbol,buy_sl,buy_tp)
                            last_order_time[symbol] = current_time
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                logging.info(f'{symbol} BUY order placed successfully with SL: {buy_sl} and TP: {buy_tp}')
                                log_trade(symbol, result.order, 'Uptrend with white soldier',buy_sl,buy_tp,'TBD',log_filename)
                            else:
                                logging.error(f'{symbol} failed to place BUY order with error: {result.retcode}')
                                log_trade(symbol, result.order, 'Uptrend detected with white soldier pattern', f'Failed with error code: {result.retcode}',log_filename)

                elif current_trend == 'bearish':
                    print(f'{symbol} is trending bearish.')
                    if (ask_price < vwap_sell) and (latest_rsi < 40):
                        print(f'{symbol} volume is bearish.')
                        if (df_15min['%K'].iloc[-1] < 30) and (df_15min['%D'].iloc[-1] < 30) and stoch_crossover['Sell_signal'].iloc[-1] == 1:
                            print(f'{symbol} stochastic crossover bearish.')
                            result = place_sell_order(lot_size,ask_price,symbol,sell_sl,sell_tp)
                            last_order_time[symbol] = current_time
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                logging.info(f'{symbol} SELL order placed successfully with SL: {sell_sl} and TP:{sell_tp}')
                                log_trade(symbol,result.order,'Downtrend with black crow',sell_sl,sell_tp,'TBD',log_filename)
                                
                            else:
                                logging.error(f'{symbol} failed to place SELL order with error: {result.retcode}')
                                log_trade(symbol, result.order, 'Downtrend detected with black crow pattern', f'failed with error code:{result.retcode}',log_filename)
                                    
                else:
                    print(f'{symbol} is trending neutral.')
                    
            except Exception as e:
                logging.error('Error occurred while placing the order: %s',e)
                
        pip_threshold = 10 * mt5.symbol_info(symbol).point
        
                
#         try:
#             positions = mt5.positions_get()
#             tick = mt5.symbol_info_tick(symbol)
#             for position in positions:
#                 if position.symbol in symbols:
#                     symbol = position.symbol
#                     order_ticket = position.ticket
#                     if position.type == mt5.ORDER_TYPE_BUY:
#                         current_price = mt5.symbol_info_tick(symbol).bid
#                         entry_price = position.price_open
#                         if current_price - entry_price >= pip_threshold:
#                             new_sl = mt5.symbol_info_tick(symbol).bid - trailing_stop_value * mt5.symbol_info(symbol).point
#                             print('buy',symbol,new_sl,position.sl)
#                             if new_sl > position.sl:
#                                 modify_stop_loss(symbol,order_ticket,new_sl)
#                                 logging.info(f'Trailing stop adjusted for BUY position on {symbol} from {position.sl} to New SL:{new_sl}')
#                                 log_trade(symbol,order_ticket,'BUY','Trailing stop adjusted.',new_sl,position.tp,'TBD',log_filename)
                            
#                     elif position.type == mt5.ORDER_TYPE_SELL:
#                         current_price = mt5.symbol_info(symbol).ask
#                         entry_price = position.price_open
#                         if current_price - entry_price >= pip_threshold:    
#                             new_sl = mt5.symbol_info_tick(symbol).ask + trailing_stop_value * mt5.symbol_info(symbol).point
#                             print('sell',symbol,new_sl,position.sl)
#                             if new_sl < position.sl:
#                                 modify_stop_loss(symbol,order_ticket,new_sl)
                                
#                                 logging.info(f'Trailing stop adjusted for SELL position on {symbol} from {position.sl} to New SL:{new_sl}')
#                                 log_trade(symbol,order_ticket,'SELL','Trailing stop adjusted.',new_sl,position.tp,'TBD',log_filename)
                                
#         except Exception as e:
#             logging.error('Error occurred while adjusting trailing stop: %s',e)
            
#         for position in positions:
#             if position.symbol in symbols:
#                 symbol = position.symbol
#                 order_ticket = position.ticket
#                 result = 'profit' if position.profit > 0 else 'Loss'
#                 update_trade_result(log_filename,order_ticket,result)
                
            
#         time.sleep(3)        
        
except KeyboardInterrupt:
    logging.info('Program interrupted by User.')
    mt5.shutdown()