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


account_number = 208016707
# account_number = 180299878
account_password = 'Cliffnonu@2'
account_server = 'Exness-MT5Trial9'

client = MetaTrader5Client(account_number,account_password,account_server)
client.login()


symbols = [
           'EURUSDm','EURJPYm','EURAUDm','USDCADm', 
           'USDJPYm','GBPUSDm','AUDUSDm','USDCADm','NZDUSDm',
           'USDCHFm','EURGBPm']

order_interval = 10800
trailing_stop_value = 1
risk_percent = 1
# lot_size = 0.01
log_filename = r'C:\Users\Administrator\work\forexBot\core\logs\forex_trades.log'
initialise_log_file(log_filename)

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler(log_filename),logging.StreamHandler()])

cycle_count = 0 
last_order_time = {symbol:0 for symbol in symbols}
start_time = pd.Timestamp.now(tz='Africa/Nairobi').replace(hour=6, minute=0)
end_time = pd.Timestamp.now(tz='Africa/Nairobi').replace(hour=18, minute=0)
time_utc = pd.Timestamp.now('Africa/Nairobi')

try:
    while True:
        if start_time < time_utc < end_time:
            print(start_time)
            print(end_time)
            print(time_utc)
        
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
                
                stochastic = Indicators.calculate_stochastic_with_slowing(df_five_min)
                # print(stochastic)
                
                stoch_crossover = Indicators.detect_stoch_crossover(stochastic)
                # print('stoch_crossover',stoch_crossover)
                
                atr = calculate_atr(df_one_hour)
                sell_sl,sell_tp = calculate_sell_stop_loss_take_profit(ask_price,atr)
                buy_sl,buy_tp = calculate_buy_stop_loss_take_profit(bid_price,atr)
                four_hour_ema = EMA(df_four_hour)
                one_hour_ema = EMA(df_one_hour)
                four_hourly_trend = predict_trend(four_hour_ema)
                one_hourly_trend = predict_trend(one_hour_ema)
                current_four_trend = four_hourly_trend[-1]
                current_one_trend = one_hourly_trend[-1]
                # print(hourly_ema)
                # print(hourly_trend[-1])
                rsi = Indicators.calculate_rsi(df_one_hour['close'])
                latest_rsi = rsi.iloc[-1]
                print('latest_rsi',latest_rsi)
                ichimoku = Indicators.ichimoku_strategy(df_one_hour)
                current_ichimoku = ichimoku.iloc[-1]
                average_volume = df_15min['tick_volume'].rolling(window=20).mean().iloc[-1]
                current_volume = df_15min['tick_volume'].iloc[-1]
                
                try:
                    
                    try:
                        # try:
                        #     exit_on_volume = current_volume < average_volume * 0.7
                        #     if exit_on_volume:
                        #         close
                                
                            if current_four_trend == 'bullish' and current_one_trend == 'bullish':
                                print(f'{symbol} is trending bullish.')  
                                # if (df_15min['%K'].iloc[-1] > 70) and (df_15min['%D'].iloc[-1] > 70) and stoch_crossover['Buy_signal'].iloc[-1] == 1:
                                if stoch_crossover.iloc[-1]['Buy_signal'] == 1:
                                    print(f'{symbol} stochastic crossover bullish.')
                                    result = place_buy_order(lot_size,bid_price,symbol,buy_sl,buy_tp)
                                    last_order_time[symbol] = current_time
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        logging.info(f'{symbol} 2:BUY order placed successfully with SL: {buy_sl} and TP: {buy_tp}')
                                        log_trade(symbol, result.order, 'Uptrend with Stochastic signal 2',buy_sl,buy_tp,'TBD',log_filename)
                                    else:
                                        logging.error(f'{symbol} failed to place BUY order with error: {result.retcode}')
                                        log_trade(symbol, result.order, 'Uptrend detected with stochastic signal', f'Failed with error code: {result.retcode}',log_filename)
                            elif current_four_trend == 'bearish' and current_one_trend == 'bearish':
                                print(f'{symbol} is trending bearish.')
                                # if (df_15min['%K'].iloc[-1] < 30) and (df_15min['%D'].iloc[-1] < 30) and stoch_crossover['Sell_signal'].iloc[-1] == 1:
                                if stoch_crossover.iloc[-1]['Sell_signal'] == 1:
                                    print(f'{symbol} 2:stochastic crossover bearish.')
                                    result = place_sell_order(lot_size,ask_price,symbol,sell_sl,sell_tp)
                                    last_order_time[symbol] = current_time
                                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                                        logging.info(f'{symbol} 2:SELL order placed successfully with SL: {sell_sl} and TP:{sell_tp}')
                                        log_trade(symbol,result.order,'Downtrend with Stochastic signal 2',sell_sl,sell_tp,'TBD',log_filename)
                                        
                                    else:
                                        logging.error(f'{symbol} 2:failed to place SELL order with error: {result.retcode}')
                                        log_trade(symbol, result.order, '2:Downtrend detected with Stochastic signal', f'failed with error code:{result.retcode}',log_filename)
                                            
                            else:
                                print(f'{symbol} is trending neutral.')
                        
                            
                    except Exception as e:
                        logging.error('2:Error occurred while placing the order: %s',e)
                        
                # except (mt5.ConnectionError, mt5.NetworkError):
                except Exception as e:
                    logging.error(f'2:Connection to MetaTrader 5 server lost.{e}')
                    time.sleep(5)
                    client.login()
            if symbol_info is None:
                print(f'Error: SYmbol{symbol} is not available in market watch')
            else:
                pip_threshold = 10 * mt5.symbol_info(symbol).point
            
                    
            try:
                positions = mt5.positions_get()
                tick = mt5.symbol_info_tick(symbol)
                for position in positions:
                    if position.symbol in symbols:
                        symbol = position.symbol
                        order_ticket = position.ticket
                        if position.type == mt5.ORDER_TYPE_BUY:
                            current_price = mt5.symbol_info_tick(symbol).bid
                            point = mt5.symbol_info(symbol).point
                            entry_price = position.price_open
                            trail_buy_amt = trailing_stop_value * mt5.symbol_info(symbol).point
                            if current_price - entry_price >= pip_threshold:
                                new_sl = position.sl - trail_buy_amt
                                print('buy',symbol,new_sl,position.sl)
                                if new_sl > position.sl:
                                    modify_stop_loss(symbol,order_ticket,new_sl)
                                    logging.info(f'Trailing stop adjusted for BUY position on {symbol} from {position.sl} to New SL:{new_sl}')
                                    log_trade(symbol,order_ticket,'BUY','Trailing stop adjusted.',new_sl,position.tp,'TBD',log_filename)
                                
                        elif position.type == mt5.ORDER_TYPE_SELL:
                            current_price = mt5.symbol_info(symbol).ask
                            entry_price = position.price_open
                            trail_sell_amt = trailing_stop_value * mt5.symbol_info(symbol).point
                            if current_price - entry_price >= pip_threshold:    
                                new_sl = position.sl + trail_sell_amt
                                print('sell',symbol,new_sl,position.sl)
                                if new_sl < position.sl:
                                    modify_stop_loss(symbol,order_ticket,new_sl)
                                    
                                    logging.info(f'Trailing stop adjusted for SELL position on {symbol} from {position.sl} to New SL:{new_sl}')
                                    log_trade(symbol,order_ticket,'SELL','Trailing stop adjusted.',new_sl,position.tp,'TBD',log_filename)
                                    
            except Exception as e:
                logging.error('Error occurred while adjusting trailing stop: %s',e)
                
            for position in positions:
                if position.symbol in symbols:
                    symbol = position.symbol
                    order_ticket = position.ticket
                    result = 'profit' if position.profit > 0 else 'Loss'
                    update_trade_result(log_filename,order_ticket,result)
                    
                
            time.sleep(3)        
        else:
            print("Skipping trade outside peak hours.")
except KeyboardInterrupt:
    logging.info('Program interrupted by User.')
    mt5.shutdown()