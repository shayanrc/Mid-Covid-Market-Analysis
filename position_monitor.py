import datetime as dt
import pandas as pd

from alice_blue import AliceBlue, LiveFeedType
from pushbullet import PushBullet

import time
import toml

socket_opened = False
positions_ltp = None
expiry_last_high = {}

trailing_stop_callback_rate = 0.02

def on_open_callback():
    global socket_opened
    
    socket_opened = True


def on_tick(update):
    # print(f"Update {update}")

    global positions_ltp
    
    
    symbol = update['instrument'].symbol
    token = update['token']
    ltt = update['ltt']
    last_trade_time = dt.datetime.fromtimestamp(ltt)
    ltp = update['ltp']
    ltq = update['ltq']
    positions_ltp[str(token)]=ltp

    print(f'{token} update: {last_trade_time} : {symbol} : {ltp} - {ltq}')
        
        
def on_order(order):
    print('ORDER UPDATE')
    print(order)


def on_market_status_message(msg):
    print('MARKET MSG')
    print(msg)
    

def evaluate_positions_by_expiry(positions, target_gains_percent = 10):
    target_rate = 1+(target_gains_percent/100)
    positions['value'] = positions['last_price'] * positions['qty']
    positions_by_expiry = positions.groupby(['underlying', 'expiry']).agg(qty=pd.NamedAgg(column="qty", aggfunc="sum"),
                                                                        lot_size=pd.NamedAgg(column="lot_size", aggfunc="first"),
                                                                        current_value=pd.NamedAgg(column="value", aggfunc="sum"),
                                                                        # current_price=pd.NamedAgg(column="last_price", aggfunc="sum"),
                                                                        last_update=pd.NamedAgg(column="last_update", aggfunc="max"),
                                                                        initial_investment=pd.NamedAgg(column="cost", aggfunc="sum") ).reset_index()

    positions_by_expiry['num_pairs'] = positions_by_expiry.qty/positions_by_expiry.lot_size/2
    positions_by_expiry['breakeven'] = positions_by_expiry['initial_investment']/(positions_by_expiry['qty']/2)
    positions_by_expiry['target'] = target_rate*positions_by_expiry['breakeven']

    # positions_by_expiry['target'] = target_rate*positions_by_expiry['initial_investment']/(positions_by_expiry['qty']/2)

    positions_by_expiry['change'] = (positions_by_expiry['current_value']-positions_by_expiry['initial_investment'])/positions_by_expiry['initial_investment']
    positions_by_expiry['position_price'] = positions_by_expiry['current_value']/(positions_by_expiry['qty']/2)

    return positions_by_expiry

def main():
    global positions_ltp
    config_toml = toml.load('.gcloud/config.toml')
    alice_blue_cred = config_toml['aliceblue']
    pb_token = config_toml['pushbullet']['access_token']
    
    pb = PushBullet(pb_token)


    print('requesting aliceblue token')
    access_token = AliceBlue.login_and_get_access_token(username=alice_blue_cred['username'], 
                                                        password=alice_blue_cred['password'], 
                                                        twoFA=alice_blue_cred['twoFA'], 
                                                        api_secret=alice_blue_cred['api_secret'], 
                                                        app_id=alice_blue_cred['app_id'])
    print(f'recieved access token {access_token}')
    alice = AliceBlue(username=alice_blue_cred['username'], 
                    password=alice_blue_cred['password'], 
                    access_token=access_token, 
                    master_contracts_to_download=['NSE', 'NFO'])

    indices = ['NIFTY', 'BANKNIFTY']

    # Retrieve all positions from broker
    positions_resp = alice.get_netwise_positions()
    if positions_resp['status']=='success':
        # positions = positions_resp['data']['positions']
        positions_df = pd.DataFrame(positions_resp['data']['positions'])[['trading_symbol', 'net_quantity', 'exchange', 'net_amount', 'ltp', 'average_buy_price', 'strike_price', 'instrument_token','unrealised_pnl']].copy()
        
        filter_exchange = filter(lambda x: x['exchange']=='NFO', positions_resp['data']['positions'])

        # make a globally accesible dict to store
        positions_ltp = dict(map(lambda x: (x['instrument_token'],x['ltp']), # token and price kv pairs
                            filter_exchange) # after filtering out on exchange
                        )
    else:
        positions_df = None
    
    
    print(positions_ltp)
    # Filter relevant columns
    # positions_df = positions[['trading_symbol', 'net_quantity', 'exchange', 'net_amount', 'ltp', 'average_buy_price', 'strike_price', 'instrument_token','unrealised_pnl']].copy()
    positions_df = positions_df[positions_df['exchange']=='NFO']
    positions_df.average_buy_price = positions_df.average_buy_price.astype(float)
    positions_df.net_quantity = positions_df.net_quantity.astype(int)

    positions_df['breakeven'] = (positions_df.average_buy_price*positions_df.net_quantity).sum()/positions_df.net_quantity
    positions_df['underlying'] = ''
    positions_df['strike_price'] = 0.0
    positions_df['option_type'] = ''
    positions_df['symbol'] = ''
    positions_df['lot_size'] = 0
    positions_df['expiry'] = None
    positions_df['last_update'] = dt.datetime.now()
    instruments = []

    for ix, position in positions_df.iterrows():
        instrument = alice.get_instrument_by_token(position['exchange'], position['instrument_token'])
        instruments.append(instrument)
        # print(instrument)
        sym_list = instrument.symbol.split()
        positions_df.loc[ix, 'underlying'] = sym_list[0]
        positions_df.loc[ix, 'strike_price'] = sym_list[3]
        positions_df.loc[ix, 'option_type'] = sym_list[4]
        positions_df.loc[ix, 'symbol'] = instrument.symbol
        positions_df.loc[ix, 'lot_size'] = instrument.lot_size
        positions_df.loc[ix, 'expiry'] = instrument.expiry

    positions_df['cost'] = -positions_df['net_amount']

    positions_df = positions_df.rename({'net_quantity':'qty',
                                        'average_buy_price':'avg_price',
                                        'ltp':'last_price'}, 
                                        axis=1)
    
    positions_df['value'] = positions_df['last_price']*positions_df['qty']
    positions_df.lot_size = positions_df.lot_size.astype(int)
    
    alice.start_websocket(subscribe_callback=on_tick,
                          socket_open_callback=on_open_callback,
                          order_update_callback=on_order,
                          socket_close_callback=None, 
                          socket_error_callback=None, 
                          run_in_background=True, 
                          market_status_messages_callback=on_market_status_message, 
                          exchange_messages_callback=None, 
                          oi_callback=None, 
                          dpr_callback=None)

    while socket_opened==False:
        print("waiting for websockets to open")
        time.sleep(1)

    
    
    print(f"Socket open")
    alice.subscribe(instruments, LiveFeedType.MARKET_DATA)
    # alice.subscribe(indices, LiveFeedType.MARKET_DATA)
    print(f"subscribed to : {alice.get_all_subscriptions()}")
    # positions = positions_df.to_dict(orient='records')
    # Continuously monitor 
    while True:
        time.sleep(5)
        print(positions_ltp)
        for token, price in positions_ltp.items():
            positions_df.loc[positions_df['instrument_token']==token, 'last_price'] = price
        pos_by_ex = evaluate_positions_by_expiry(positions_df)
        print(pos_by_ex)

        sell_candidates = pos_by_ex[pos_by_ex['position_price']>=pos_by_ex['target']]
        
        if len(sell_candidates)>0 or len(expiry_last_high)>0:
            print(expiry_last_high)
            for ix, row in pos_by_ex.iterrows():
                pos_name = f"{row['underlying']} - {row['expiry']}"

                # If a new high above target has been achieved
                if pos_name not in expiry_last_high.keys():
                    # save the high 
                    expiry_last_high[pos_name]= row['position_price']
                    # and send a notification
                    pb.push_note(pos_name, f"Position Price : Rs. {row['position_price']}\n Target: Rs. {row['target']} ")
                else:
                    # If a new high above previous high has been achieved
                    if row['position_price']>expiry_last_high[pos_name]:
                        # update new high
                        expiry_last_high[pos_name] = row['position_price']
                    # if it has dropped by the callback rate since last high
                    elif row['position_price'] < expiry_last_high[pos_name]*(1-trailing_stop_callback_rate):
                        # send a sell notification
                        pb.push_note(f"SELL {pos_name}", 
                                     f"Position Price : Rs. {row['position_price']}\n Last high: Rs. {expiry_last_high[pos_name]} ")
                        

                



    
if __name__=='__main__':
    main()
    

