import matplotlib.pyplot as plt

import seaborn as sns

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os

import datetime as dt
from dateutil import relativedelta
import nsepy

import requests
import json

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(
   page_title="Optimal Options",
   initial_sidebar_state="expanded",
)
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),'datastore/processed')
DATA_ROOT = 'QmVmo99nn2wMjShNGi5CDPEqzRWtYxWRrjD5XTsfQiCZD3'
# DATA_ROOT = 'QmYcxeeDVJzaHGwBxJNnZ66hbr9rV8BYMbmBHZxsUfWabH'

def load_ohlcv_files(csv_folder):
    csv_files = glob.glob(csv_folder+'/*.csv',)
    ohlcv_df = pd.concat([pd.read_csv(f, parse_dates=['Date'], index_col=['Date']) for f in csv_files], 
                        join='outer')

    ohlcv_df = ohlcv_df.drop_duplicates()

    return ohlcv_df


def round_to_base(x, base=50):
    return base * round(x/base)


@st.cache
def calculate_period_delta(ohlc_df, period):

    if period > 1:

        period_change = ohlc_df.Close.rolling(period).apply(lambda x: (x[-1]-x[0])/x[0]).dropna()

        period_open = ohlc_df.Open.rolling(period).apply(lambda x: x[0]).dropna()

        period_high = ohlc_df.High.rolling(period).apply(lambda x: x.max()).dropna()

        period_low = ohlc_df.Low.rolling(period).apply(lambda x: x.min()).dropna()
    
    else:

        period_change = ohlc_df.Close - ohlc_df.Open
        period_open = ohlc_df.Open
        period_high = ohlc_df.High
        period_low = ohlc_df.Low


    period_pos_change = (period_high - period_open)/period_open
    period_neg_change = (period_low - period_open)/period_open

    period_delta_df = pd.DataFrame({'delta':period_change, 'delta_max':period_pos_change, 'delta_min':period_neg_change}, )

    return period_delta_df

def get_last_working_day(today=None):
    if today is None:
        today = dt.date.today()
    last_working_day = today-dt.timedelta(days=1)
    while not np.is_busday(last_working_day):
        last_working_day = last_working_day-dt.timedelta(days=1)
    return last_working_day

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

def evaluate_postions(options_data, underlying_period_delta, underlying_price, capital=10000, expected_rate=0.1, lot_size=50, otm_only=True):
    # Separate out Call and Put option
    ce_df = options_data[options_data['Option Type']=='CE']
    pe_df = options_data[options_data['Option Type']=='PE']

    # Filter out ITM
    ce_df = ce_df[ce_df['Strike Price']>=underlying_price]
    pe_df = pe_df[pe_df['Strike Price']<=underlying_price]

    # Sort by Strike
    ce_df = ce_df.sort_values('Strike Price')
    pe_df = pe_df.sort_values('Strike Price')

    # Filter out options we can't invest in due to capital constraints
    ce_df = ce_df[(ce_df['Close']*lot_size)<capital]
    pe_df = pe_df[(pe_df['Close']*lot_size)<capital]
    

    # flat_brokerage = 20 # placeholder in Rs
    # capital_breakeven = (capital*(1+expected_rate)+tax)/lot_size
    capital_breakeven = ((1+expected_rate)*capital/lot_size)

    # Maximum Change required to break even on position
    ce_df['change_breakeven'] = ((ce_df['Strike Price'] + ce_df['Close'] - underlying_price)/underlying_price).values
    pe_df['change_breakeven'] = ((pe_df['Strike Price'] - pe_df['Close'] - underlying_price)/underlying_price).values

    # Change percent required to close ITM
    ce_df['change_itm'] = ((ce_df['Strike Price'] - underlying_price)/underlying_price).values
    pe_df['change_itm'] = ((pe_df['Strike Price'] - underlying_price)/underlying_price).values

    # Position cost per lot
    ce_df['position_cost'] = lot_size*ce_df['Close']
    pe_df['position_cost'] = lot_size*pe_df['Close']
    
    
    delta_mean = underlying_period_delta['delta'].mean()
    delta_std = underlying_period_delta['delta'].std()
    
    # Calculate underlying price after sigma change in either direction
    sigma_high = underlying_price*(1+delta_mean+delta_std)
    sigma_low = underlying_price*(1+delta_mean-delta_std)
    
    # Profits when underlying expires at mean +/- std
    ce_df['sigma_returns'] = lot_size*(sigma_high - ce_df['Strike Price'])
    pe_df['sigma_returns'] = lot_size*(pe_df['Strike Price'] - sigma_low)
    
    ce_df['sigma_target'] = ce_df['sigma_returns']/lot_size
    pe_df['sigma_target'] = pe_df['sigma_returns']/lot_size
    
    # Probabilty of breaking even within the period
    pe_df['probability_breakeven'] = pe_df.change_breakeven.apply(lambda x:  (underlying_period_delta.delta_min<x).mean())
    ce_df['probability_breakeven'] = ce_df.change_breakeven.apply(lambda x:  (underlying_period_delta.delta_max>x).mean())
    
    # Probability of closing in the money
    pe_df['probability_itm'] = pe_df.change_itm.apply(lambda x:  (x>underlying_period_delta.delta).mean())
    ce_df['probability_itm'] = ce_df.change_itm.apply(lambda x:  (x<underlying_period_delta.delta).mean()) 
    
    # Filter out useless columns
    ce_df = ce_df[['Symbol',
               'Expiry', 
               'Option Type', 
               'Strike Price', 
               'Close', 
               'change_breakeven', 
               'change_itm', 
               'position_cost', 
                'sigma_returns', 
                'sigma_target',
               'probability_breakeven', 
               'probability_itm']]

    pe_df = pe_df[['Symbol', 
                   'Expiry', 
                   'Option Type', 
                   'Strike Price', 
                   'Close', 
                   'change_breakeven', 
                   'change_itm', 
                   'position_cost', 
                   'sigma_returns', 
                   'sigma_target',
                   'probability_breakeven', 
                   'probability_itm']]
    
    return ce_df, pe_df

def generate_pairs(pe_positions, ce_positions, fitler_sigma=True, brokerage=20):
    
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


def evaluate_pairs(position_pairs, underlying_close, delta_df, lot_size=50, expected_rate=0.05):
    # Returns required to cover expected losses
    # required_returns = position_pairs['pair_otm_probability']/(1-position_pairs['pair_otm_probability'])
    # required_returns = (position_pairs['cost']*(1+required_returns))
    required_returns = (position_pairs['cost']*(1+expected_rate))
    position_pairs['target'] = required_returns/lot_size
    
    # Change requrired to breakeven
    pe_breakeven_change = (position_pairs['pe_strike_price'] - position_pairs['target']) -underlying_close
    ce_breakeven_change = (position_pairs['ce_strike_price'] + position_pairs['target']) -underlying_close

    position_pairs['pe_breakeven_change'] = pe_breakeven_change
    position_pairs['ce_breakeven_change'] = ce_breakeven_change

    # Probabilty of breaking even due to valatility within the period
    pe_breakeven_probability = pe_breakeven_change.apply(lambda x:  (delta_df.delta_min<(x/underlying_close)).mean())
    ce_breakeven_probability = ce_breakeven_change.apply(lambda x:  (delta_df.delta_max>(x/underlying_close)).mean())
    position_pairs['pe_breakeven_probability'] = pe_breakeven_probability
    position_pairs['ce_breakeven_probability'] = ce_breakeven_probability
    position_pairs['pair_not_breakeven_probability'] = (1-pe_breakeven_probability)*(1-ce_breakeven_probability)
    position_pairs = position_pairs.sort_values('pair_not_breakeven_probability',  ignore_index=True)
    
    return position_pairs

@st.cache(allow_output_mutation=True, ttl=100)
def load_options_data(underlying, is_index=True):
    
    if is_index:
        nse_url = 'https://www.nseindia.com/api/option-chain-indices'
    else:
        nse_url = 'https://www.nseindia.com/api/option-chain-equities'

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
    
    resp = requests.get(nse_url, params={'symbol':underlying}, headers=headers, cookies=cookies)
    # Load into a dataframe
    resp_json = json.loads(resp.text)
    # print(resp_json)
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
    return option_df, expiries

# PROVIDER_URL = 'https://ipfs.infura.io/ipfs/'
PROVIDER_URL = 'https://cloudflare-ipfs.com/ipfs/'

@st.cache(allow_output_mutation=True, ttl=3600*12, persist=True)
def load_underlying(underlying, data_root=DATA_ROOT):
    
    data_dict = requests.get(PROVIDER_URL+data_root).json()
    if underlying not in data_dict.keys():
        underlying_df = nsepy.get_history(underlying, dt.datetime.today()-dt.timedelta(days=365*3), dt.datetime.today())
        
    else:
        data_hash = data_dict[underlying]        
        underlying_df = pd.read_json(PROVIDER_URL+data_hash)
        # print(underlying_df.head())
        underlying_df.loc[:, 'Date'] = pd.to_datetime(underlying_df['Date'])
        underlying_df = underlying_df.set_index('Date')

    return underlying_df

# @st.cache(ttl=3600*24*7, persist=True)
def get_lotsize(underlying, is_index=True):
    if is_index:
        inst_type = 'FUTIDX'
    else:
        inst_type = 'FUTSTK'
    
    expiry = dt.date.today()+relativedelta.relativedelta(day=31, weekday=relativedelta.TH(-1))
    
    if expiry<dt.date.today():
        expiry = (dt.date.today()+dt.timedelta(days=7))+relativedelta.relativedelta(day=31, weekday=relativedelta.TH(-1))
    
    print(expiry)
    contract_quote = nsepy.get_quote(symbol=underlying, series='EQ', instrument=inst_type, expiry=expiry)
    print(contract_quote)
    market_lot = contract_quote['data'][0]['marketLot']    
    return market_lot

st.title('Options Strategy Probability Optimization')

# instruments  = ['NIFTY', 'BANKNIFTY', 'DLF']
indices = ['NIFTY', 'BANKNIFTY']
# instruments = [x for x in instruments if not x.startswith('.')]

with open('/mnt/d/Workspace/Mid-Covid-Market-Analysis/datastore/nse.dataroot', mode='r') as f:
    # data_dict = requests.get(PROVIDER_URL+f.readline()).json()
    # data_hash = data_dict['NIFTY']
    data_root = f.readline()
    data_dict = requests.get(PROVIDER_URL+data_root).json()
    instruments = list(data_dict.keys())

sidebar_form = st.sidebar.form(key='sidebar_form')
underlying = sidebar_form.selectbox('Underlying', instruments)

st.header('%s' % underlying)
#Load Underlying Data
# underlying_path = os.path.join(DATA_PATH, underlying)
# ohlc_path = os.path.join(underlying_path, 'OHLC')
# ohlc_path = os.path.join(underlying_path, underlying+'.csv')
# underlying_df = load_ohlcv_files(ohlc_path)
# underlying_df = pd.read_csv(ohlc_path, parse_dates=['Date'], index_col=['Date'])
# underlying_df = pd.read_json('https://bafybeieoritg3qozjxewiamkin2ded7vri6j55w33kpljfvhciul5njsku.ipfs.infura-ipfs.io/')
# underlying_df = load_underlying(underlying)

    # data_dict = requests.get(PROVIDER_URL+f.readline()).json()
    # print(data_dict.keys())
    # underlying_df = pd.read_json(PROVIDER_URL+data_hash)


underlying_df = load_underlying(underlying, data_root=data_root)
underlying_df
with st.expander('Historical data details') as underlying_upload_expander:
    st.write('Earliest Historical data available:')
    st.write(underlying_df.index.min())
    st.write('Latest Historical data available:')
    st.write(underlying_df.index.max())

    underlying_df.head()


# Upload underlying Data
# with st.expander('Upload More Data') as underlying_upload_expander:
#     underlying_upload_md = """Download Historical OHLC data for indexes from here: 
#                             https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm
#                             And upload it below to add it to your anylisis 
#                             """
#     uploaded_file = st.file_uploader("Upload Underlying Data", type="csv", help=underlying_upload_md, key='underlying_uploader')
#     if uploaded_file is not None:
#         underlying_new = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col=['Date'])
#         st.write('New data uploaded for:')
#         st.write('Earliest:')
#         st.write(underlying_new.index.min())
#         st.write('Latest:')
#         st.write(underlying_new.index.max())

today = dt.date.today()
#Get all expiries this month
# expiries = nsepy.get_expiry_date(today.year, today.month, index=True)
# expiries = list(expiries)
# And keep only future expiris
# expiries = [expiry for expiry in expiries if expiry>today]

is_index = underlying in indices
option_df, expiries = load_options_data(underlying, is_index=is_index)
# if len(expiries)==0:
#     expiries = [dt.date.today()]
# trade_days_left = np.busday_count(today, expiries)
last_close = option_df['Underlying'].values[0]
if last_close == 0:
    last_close = underlying_df.Close.iloc[-1]
# st.write(last_close)
last_close = sidebar_form.number_input('Last Close', min_value=round_to_base(last_close*0.9, base=10), max_value=round_to_base(last_close*1.1, base=10), value=int(last_close), step=10)

expiry = sidebar_form.selectbox('Choose Expiry', expiries)
# calc_trade_days_left = trade_days_left[expiries.index(expiry)]

# calc_trade_days_left = 5
calc_trade_days_left = np.busday_count(today, dt.datetime.strptime(expiry, '%d-%b-%Y').date())
# st.text('Selected expiry %s'%expiry)
# st.write('Trade Days Left to Expiry:' , calc_trade_days_left)
st.markdown('## Historical Change Analysis')   

trade_days_left = sidebar_form.slider('Adjust Trade Days Left to Expiry', 1, 32, int(calc_trade_days_left))

underlying_period_delta = calculate_period_delta(underlying_df, trade_days_left)


fig = make_subplots(rows=3, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.1, 
                    subplot_titles = ['Underlying', 'Trend', 'Computed Volatility'])

fig.append_trace(go.Candlestick(x=underlying_df.index, 
                                open=underlying_df['Open'], 
                                high=underlying_df['High'], 
                                low=underlying_df['Low'], 
                                close=underlying_df['Close']),
                 row=1, col=1)
fig.append_trace(go.Scatter(x=underlying_df.index, 
                            y=underlying_period_delta['delta'], 
                            fill='tozeroy',
                            mode='none'),
                 row=2, col=1)
fig.append_trace(go.Bar(x=underlying_df.index, 
                        y=underlying_period_delta['delta_max']-underlying_period_delta['delta_min']),
                 row=3, col=1)
fig.append_trace(go.Bar(x=underlying_df.index, 
                        y=abs(underlying_period_delta['delta'])),
                 row=3, col=1)
# fig = go.Figure(data=[go.Candlestick(x=underlying_df.index,
#                 open=underlying_df['Open'], high=underlying_df['High'],
#                 low=underlying_df['Low'], close=underlying_df['Close'])
#                       ])
# fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    title='%s'%underlying,
    xaxis_rangeslider_visible=False,
    showlegend = False,
    # yaxis_title='Daily',
    # shapes = [dict(
    #     x0='2016-12-09', x1='2016-12-09', y0=0, y1=1, xref='x', yref='paper',
    #     line_width=2)],
    # annotations=[dict(
    #     x='2016-12-09', y=0.05, xref='x', yref='paper',
    #     showarrow=False, xanchor='left', text='Increase Period Begins')]
    margin=dict(l=10, r=0, t=25, b=0)
)
fig.update_xaxes(rangeslider= {'visible':True}, row=3, col=1)

st.plotly_chart(fig)

mean_delta = underlying_period_delta['delta'].mean()
std_delta = underlying_period_delta['delta'].std()

ax = sns.displot(data=underlying_period_delta['delta']*100,
                kde=True, 
                aspect=2.5)
ax.set_axis_labels(x_var='% Change in underlying')
plt.grid(axis='y')
plt.axvline(x=100*mean_delta, 
            # color='blue', 
            linewidth=1.0, 
            linestyle='-')
plt.axvline(x=100*(mean_delta-std_delta), 
            color='green', 
            linewidth=1.0, 
            linestyle='--')
plt.axvline(x=100*(mean_delta+std_delta), 
            color='orange', 
            linewidth=1.0, 
            linestyle='--')
st.pyplot(ax)

expected_change_neg_sigma = mean_delta - std_delta
prob_change_neg_sigma = (underlying_period_delta['delta']<expected_change_neg_sigma).sum()/len(underlying_period_delta['delta'])
underlying_sigma_minus = last_close*(expected_change_neg_sigma+1)
expected_change_pos_sigma = mean_delta + std_delta
prob_change_pos_sigma = (underlying_period_delta['delta']>expected_change_pos_sigma).sum()/len(underlying_period_delta['delta'])
underlying_sigma_plus = last_close*(expected_change_pos_sigma+1)


st.markdown('### Probabilty of 1 Standard Deviation Change in %s'%underlying)   
with st.expander('Probability Formula'):
    st.latex(r"P(\Delta U_{p+} > \overline{\Delta U} + \sigma)")
st.write("%.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_neg_sigma, underlying_sigma_minus, expected_change_neg_sigma*100))

st.write("%.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_pos_sigma, underlying_sigma_plus, expected_change_pos_sigma*100))


with st.container() as cntnr:
    st.markdown("### Change Probability Analysis")
    change_choice = st.radio('Price Or Change %', ('Price', 'Change'), index=0)
    if change_choice == 'Change':
        expected_changes = st.slider('Expected Change (%)', min_value=-10.0, max_value=10.0, value=(-1.5, 1.5), step=0.25)
        expected_high = expected_changes[1]/100
        expected_low = expected_changes[0]/100
    else:
        expected_changes = st.slider('Price Range', min_value=round_to_base(last_close*0.9, base=10), max_value=round_to_base(last_close*1.1, base=10), value=(round_to_base(last_close*1.015, 1),round_to_base(last_close*0.985, 1)), step=10)
        
        expected_high = (expected_changes[1]-last_close)/last_close
        expected_low = (expected_changes[0]-last_close)/last_close
        
    # Expiry change probability
    prob_change_plus = (underlying_period_delta['delta']>expected_high).sum()/len(underlying_period_delta['delta'])
    prob_change_max = (underlying_period_delta['delta_max']>expected_high).sum()/len(underlying_period_delta['delta'])
    underlying_change_plus = last_close*(expected_high+1)

    prob_change_minus = (underlying_period_delta['delta']<expected_low).sum()/len(underlying_period_delta['delta'])
    prob_change_min = (underlying_period_delta['delta_min']<expected_low).sum()/len(underlying_period_delta['delta'])
    underlying_change_minus = last_close*(expected_low+1)

    # Volatility change probability
    prob_change_within_range = ((underlying_period_delta['delta']>expected_low) & (underlying_period_delta['delta']<expected_high)).sum()/len(underlying_period_delta['delta'])
    prob_volatility_exceeds_range =  ((underlying_period_delta['delta_min']<expected_low) | (underlying_period_delta['delta_max']>expected_high)).sum()/len(underlying_period_delta['delta'])

    st.write("- %.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_plus, underlying_change_plus, expected_high*100))
    st.write("But %.3f probability of underlying reaching above %.2f" %(prob_change_max, underlying_change_plus))
    st.write("- %.3f probability of expiry below %.2f (%.2f%% change)" %(prob_change_minus, underlying_change_minus, expected_low*100))
    st.write("But %.3f probability of underlying going below %.2f" %(prob_change_min, underlying_change_minus))
    st.write("- %.3f probability of expiry within %.2f and %.2f" %(prob_change_within_range, underlying_change_minus, underlying_change_plus))
    st.write("- %.3f probability of price exceeding range at least once in %d trading days" %(prob_volatility_exceeds_range, trade_days_left))
    st.write("ie. %.2f %% chance of underlying reaching a price level above %.2f or below %.2f at least once before expiry" %(prob_volatility_exceeds_range*100, underlying_change_plus, underlying_change_minus))

st.header('Options Analysis')

# options_path = os.path.join(underlying_path, 'Options')

# options_path = os.path.join(underlying_path, 'Options')
# options_df = load_ohlcv_files(options_path)


# with st.expander('Upload More Data?') as option_upload_expander:

#     options_upload_md = """Download Options chain data for indexes from here: 
#                             https://www1.nseindia.com/products/content/derivatives/equities/historical_fo.htm
#                             And upload it below to add it to your anylisis 
#                             """
#     uploaded_file = st.file_uploader("Upload More Data?", type="csv", help=options_upload_md, key='options_uploader')
#     if uploaded_file is not None:
#         options_new = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col=['Date'])
#         st.write('New data uploaded for:')
#         st.write('Earliest:')
#         st.write(options_new.index.min())
#         st.write('Closest:')
#         st.write(options_new.index.max())
#         st.write(options_new['Expiry'].unique().tolist())
#         st.write(options_new['Option Type'].unique().tolist())


# Expiry Range


# Filter for latest data
# expiry = st.selectbox('Choose Expiry', options_df['Expiry'].unique())
# options_df['Expiry'] = pd.to_datetime(options_df['Expiry'])
# options_df = options_df[options_df['Expiry']==expiry]
option_df = option_df[option_df['Expiry']==expiry]
option_df['Expiry'] = pd.to_datetime(option_df['Expiry'])



# st.write('Earliest options data vailable:')
# st.write(options_df.index.min())
# st.write('Latest options data vailable:')
# st.write(options_df.index.max())

# Keep only the latest data
# options_df = options_df[options_df.index.max()==options_df.index]
# Get rid of contracts which are not actively traded

# option_df = option_df[option_df['Number of Contracts']>1000]
# option_df = option_df[option_df['Change']!=0]
option_df = option_df[option_df['Change']!=0]

with st.expander('Preview %s Options Data'% underlying) as  ohlc_preview:
    # Display Head and Tail
    st.write('Options:')
    st.dataframe(option_df.head(5))
    st.dataframe(option_df.tail(5))

capital = sidebar_form.number_input('Capital', min_value=1000, max_value=100000, value=15000, step=500)
market_lot = get_lotsize(underlying, is_index)
lot_size = sidebar_form.number_input('Lot Size', min_value=1, max_value=10000, value=int(market_lot), step=int(market_lot))
expected_rate = sidebar_form.slider('Slippage (%)', min_value=0.5, max_value=25.0, value=2.0, step=0.5)/100


sidebar_form.form_submit_button("Scan Opportunities")

# with st.expander('Add Existing Position Pairs'):
#     with st.form('existing_positions_form'):
#         put_col, call_col = st.columns(2)
#         pe_strike_prices = option_df[option_df['Option Type']=='PE']['Strike Price'].sort_values()
#         ce_strike_prices = option_df[option_df['Option Type']=='CE']['Strike Price'].sort_values()
        
#         with put_col:
#             st.write('Put Position')
#             put_price = st.number_input('Avg Buy Price', min_value=0.0, key='put_price')
#             put_strike = st.selectbox('Strike Price', pe_strike_prices, key='put_strike')
        
#         with call_col:
#             st.write('Call Position')
#             call_price = st.number_input('Avg Buy Price', min_value=0.0)
#             call_strike = st.selectbox('Strike Price', ce_strike_prices)
        
#         qty = st.number_input('Quantity', min_value=0, step=lot_size, value=lot_size)
#         add_positions = st.form_submit_button('Add Positions')
        
#         if add_positions:
#             if 'current_positions' not in st.session_state:
#                 st.session_state.current_positions = []
#             pair_cost = qty*(put_price+call_price)
#             st.session_state.current_positions.append({'pe_strike_price':put_strike, 'pe_price':put_price, 
#                                                        'ce_strike_price':call_strike, 'ce_price':call_price, 
#                                                        'cost':pair_cost})



ce_df, pe_df = evaluate_postions(options_data=option_df, underlying_period_delta=underlying_period_delta, underlying_price=last_close, capital=capital, expected_rate=expected_rate, lot_size=lot_size)
if len(ce_df)==0 or len(pe_df)==0:
    st.warning('No Viable pairs were found within capital range')
    st.stop()
period_volatility = (underlying_period_delta['delta_max'] - underlying_period_delta['delta_min']).iloc[-1]
prev_period_volatility = (underlying_period_delta['delta_max'] - underlying_period_delta['delta_min']).iloc[-2]
volatailty_decay = prev_period_volatility-period_volatility
col1, col2 = st.columns(2)
filter_volatility = col1.checkbox('Filter Volatility', value=True)
filter_sigma = col2.checkbox('Filter Sigma', value=True)

if filter_volatility:
    if volatailty_decay>0:
            
        ce_df = ce_df[ce_df['change_breakeven']<(period_volatility-volatailty_decay)]
        pe_df = pe_df[pe_df['change_breakeven'].abs()<period_volatility-volatailty_decay]
    else:    
        ce_df = ce_df[ce_df['change_breakeven']<period_volatility]
        pe_df = pe_df[pe_df['change_breakeven'].abs()<period_volatility]
if len(ce_df)==0 or len(pe_df)==0:
    st.warning('No Viable pairs were found within volitility range')
    st.stop()

with st.expander('Put Positions Analysis'):
    st.table(pe_df[['Strike Price', 'Close', 'probability_breakeven', 'probability_itm']].sort_values('probability_breakeven', ascending=False, ignore_index=True))

with st.expander('Call Positions Analysis'):
    st.table(ce_df[['Strike Price', 'Close', 'probability_breakeven', 'probability_itm']].sort_values('probability_breakeven', ascending=False, ignore_index=True))


position_pairs = generate_pairs(pe_df, ce_df, fitler_sigma=filter_sigma)
if len(position_pairs) == 0:
    st.warning('No Viable pairs were found which would breakeven at less than 1 sigma change')
    st.stop()
eval_pairs = evaluate_pairs(position_pairs, last_close, underlying_period_delta, lot_size=lot_size, expected_rate=expected_rate)



# "Volatility %s %" % period_volatility
# st.text("Trend %f %"% (underlying_period_delta['delta'].iloc[-1]))
col1, col2, col3 = st.columns(3)

col1.metric(label="Underlying", 
            value="{:.2f}".format(last_close), 
            delta="{:.2f} %".format(100*(last_close - underlying_df.Close.iloc[-1])/underlying_df.Close.iloc[-1]))
col2.metric(label="Volatility", 
            value="{:.2f} %".format(100*period_volatility), 
            delta="{:.2f} %".format(100*(period_volatility-prev_period_volatility)))

col3.metric(label="Trend", 
            value="{:.2f} %".format(100*underlying_period_delta['delta'].iloc[-1]), 
            delta="{:.2f} %".format(100*underlying_period_delta['delta'].iloc[-1]-underlying_period_delta['delta'].iloc[-2]))
st.header('Recomended Pairs')
# max(eval_pairs['ce_strike_price']-last_close, last_close-eval_pairs['pe_strike_price'])
dsp_cols = ['cost', 'ce_strike_price', 'pe_strike_price', 'ce_price', 'pe_price', 'target', 'pair_otm_probability', 'pair_not_breakeven_probability', 'spread']
# last_close - eval_pairs['pe_strike_price'], eval_pairs['ce_strike_price'] - last_close

def highlight_pairs(x):
    if x['target']>100:
        return 'background-color: red'
    # return ['background-color: red' if v == x.max() else '' for v in x]

# eval_pairs
# dsp_pairs = eval_pairs[(period_volatility*last_close)>(eval_pairs['target']+eval_pairs['spread'])]
# dsp_pairs = eval_pairs[period_volatility>(eval_pairs['ce_breakeven_change']/last_close)]
# dsp_pairs = dsp_pairs[period_volatility>abs(dsp_pairs['pe_breakeven_change']/last_close)]
dsp_pairs = eval_pairs[dsp_cols]

fig = px.scatter(dsp_pairs, x='pair_otm_probability', y='pair_not_breakeven_probability', size='spread', color='cost', hover_data=['ce_strike_price', 'pe_strike_price', 'ce_price', 'pe_price', 'target'])
x_min = dsp_pairs['pair_otm_probability'].min()
y_min = dsp_pairs['pair_not_breakeven_probability'].min()
# fig.add_annotation(
#             x=x_min,
#             y=0.5,
#             text='Long Opportunities',
#             # xref="paper",
#             # yref="paper",
#             align="right",
#             showarrow=False,
#             font_size=10
# )
# fig.add_shape(type="rect",
#     # xref="paper", yref="paper",
#     x0=x_min, y0=y_min,
#     x1=0.5, y1=0.5,
#     line=dict(
#         color="Blue",
#         width=1,
#     ),
#     opacity=0.25,
#     fillcolor="RoyalBlue",
# )
fig.add_vline(x=0.5, line_dash="dot",
              annotation_text="P(OTM @ Expiry) = 0.5", 
              annotation_position="bottom right")
fig.add_hrect(y0=y_min*0.9, y1=0.5, 
              annotation_text="Long", annotation_position="top left",
              fillcolor="green", opacity=0.25, line_width=0)
st.plotly_chart(fig)
st.dataframe(dsp_pairs)
