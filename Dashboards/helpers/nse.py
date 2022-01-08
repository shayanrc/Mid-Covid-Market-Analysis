import datetime as dt
import streamlit as st
import requests
import pandas as pd
import json
import nsepy

from dateutil import relativedelta

# PROVIDER_URL = 'https://ipfs.infura.io/ipfs/'
# PROVIDER_URL = 'https://ipfs.mihir.ch/ipfs/'
PROVIDER_URL = 'https://cloudflare-ipfs.com/ipfs/'
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
#         new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.append(('optionType', k))
            items.extend(flatten(v, k, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)

@st.cache(allow_output_mutation=True, ttl=100)
def load_options_data(underlying):
    return nse_load_options(underlying)


def nse_load_options(underlying, is_index=False):
    # Load Options Data
    # Get a copy of the default headers that requests would use
    headers = requests.utils.default_headers()

    # Update the headers with your custom ones
    headers.update(
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
        }
    )

    # Fetch the mainwebsite to get the header cookies
    resp = requests.get('https://www.nseindia.com/', headers=headers)
    cookies = resp.cookies.get_dict()
    
    if is_index:
        option_url = 'https://www.nseindia.com/api/option-chain-indices'
    else:
        option_url = 'https://www.nseindia.com/api/option-chain-equities'

    resp = requests.get(option_url, params={'symbol':underlying}, headers=headers, cookies=cookies)
    # Load into a dataframe
    resp_json = json.loads(resp.text)
    option_df = pd.DataFrame([record for record in map(flatten, resp_json['records']['data'])])
    option_df = option_df.rename({'underlying':'Symbol',
                                'expiryDate': 'Expiry',
                                'optionType': 'Option Type',
                                'strikePrice': 'Strike Price',
                                'openInterest': 'Open Interest',
                                'changeinOpenInterest': 'Change in OI',
                                'pchangeinOpenInterest': 'Change % in OI',
                                'totalTradedVolume': 'Number of Contracts',
                                'impliedVolatility': 'impliedVolatility',
                                'lastPrice': 'Close',
                                'change': 'Change',
                                'pChange': 'Change %',
                                'underlyingValue': 'Underlying'}, axis=1)
    expiries = resp_json['records']['expiryDates']
    
    expiries = pd.to_datetime(expiries)
    option_df.loc[:, 'Expiry'] = pd.to_datetime(option_df['Expiry'])

    return option_df, expiries

@st.cache(allow_output_mutation=True, ttl=3600*12, persist=True)
def load_underlying_cached(underlying, data_root):

    return load_underlying(underlying, data_root)

def load_underlying(underlying, data_root):
    # Load underlying historical data from dataroot on IPFS
    data_dict = requests.get(PROVIDER_URL+data_root).json()
    data_hash = data_dict[underlying]

    underlying_df = pd.read_json(PROVIDER_URL+data_hash)
    underlying_df.loc[:, 'Date'] = pd.to_datetime(underlying_df['Date'])
    underlying_df = underlying_df.set_index('Date')

    return underlying_df

def load_instrument_list(data_root):
    data_dict = requests.get(PROVIDER_URL+data_root).json()
    return list(data_dict.keys())

def get_lotsize(underlying, is_index=True):
    if is_index:
        inst_type = 'FUTIDX'
    else:
        inst_type = 'FUTSTK'
    
    expiry = dt.date.today()+relativedelta.relativedelta(day=31, weekday=relativedelta.TH(-1))
    contract_quote = nsepy.get_quote(symbol=underlying, series='EQ', instrument=inst_type, expiry=expiry)
    # print(contract_quote)
    market_lot = contract_quote['data'][0]['marketLot']    
    return int(market_lot)


def generate_pairs(pe_positions, ce_positions, capital, fitler_sigma=True, brokerage=20):
    
    # Convert to dict for easier iteration
    pe_positions = pe_positions.to_dict(orient='records')
    ce_positions = ce_positions.to_dict(orient='records')
    
    position_pairs = []

    for pe_position in pe_positions:

        if pe_position['sigma_returns'] <= 0 & fitler_sigma:
            continue
        for ce_position in ce_positions:
            if ce_position['sigma_returns'] <= 0 & fitler_sigma:
                continue

            pair_cost = pe_position['position_cost'] + ce_position['position_cost']
            if pair_cost < capital:
                position_pair = {'cost': pair_cost+brokerage*4}
                position_pair['pe_price'] = pe_position['Close']
                position_pair['ce_price'] = ce_position['Close']
                position_pair['capital_not_breakeven_probability'] = (1-pe_position['probability_breakeven'])*(1-ce_position['probability_breakeven'])
                position_pair['pe_strike_price'] = pe_position['Strike Price']
                position_pair['ce_strike_price'] = ce_position['Strike Price']
                position_pair['spread'] = ce_position['Strike Price'] - pe_position['Strike Price']
                position_pair['pe_sigma_target'] = pe_position['sigma_target']
                position_pair['ce_sigma_target'] = ce_position['sigma_target']
                position_pair['pe_probability_itm'] = pe_position['probability_itm']
                position_pair['ce_probability_itm'] = ce_position['probability_itm']
                position_pair['pair_otm_probability'] = (1-ce_position['probability_itm'])*(1-pe_position['probability_itm'])
                position_pairs.append(position_pair)


    position_pairs = pd.DataFrame(position_pairs)

    # position_pairs['capital_breakeven_probability'] = 1-position_pairs['capital_not_breakeven_probability']
    return position_pairs