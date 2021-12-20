
import streamlit as st
import requests
import pandas as pd
import json

PROVIDER_URL = 'https://ipfs.infura.io/ipfs/'

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


def nse_load_options(underlying):
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
    
    resp = requests.get('https://www.nseindia.com/api/option-chain-indices', params={'symbol':underlying}, headers=headers, cookies=cookies)
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
