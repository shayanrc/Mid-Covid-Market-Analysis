import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import pandas as pd
import numpy as np
import glob
import os

import datetime as dt

import math


# DATA_PATH = '../Data/'
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),'Data')
st.title('Options Strategy Probability Optimization')

st.header('Underlying')

underlying = st.selectbox('Select Underlying', os.listdir(DATA_PATH))

underlying_path = os.path.join(DATA_PATH, underlying)
ohlc_path = os.path.join(underlying_path, 'OHLC')

@st.cache
def load_ohlcv_files(csv_folder):
    csv_files = glob.glob(csv_folder+'/*.csv',)
    ohlcv_df = pd.concat([pd.read_csv(f, parse_dates=['Date'], index_col=['Date']) for f in csv_files], 
                        join='outer')

    ohlcv_df = ohlcv_df.drop_duplicates()

    return ohlcv_df

underlying_df = load_ohlcv_files(ohlc_path)

# Display Head and Tail
st.write('Earliest underlying data vailable:')
st.dataframe(underlying_df.head(5))

st.write('Latest underlying data vailable:')
st.dataframe(underlying_df.tail(5))



# Upload underlying Data

underlying_upload_expander = st.expander('Upload More Data')
underlying_upload_md = """Download Historical OHLC data for indexes from here: 
                        https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm
                        And upload it below to add it to your anylisis 
                        """
uploaded_file = underlying_upload_expander.file_uploader("Upload Underlying Data", type="csv", help=underlying_upload_md, key='underlying_uploader')
if uploaded_file is not None:
    underlying_new = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col=['Date'])
    st.write('New data uploaded for:')
    st.write('Earliest:')
    st.write(underlying_new.index.min())
    st.write('Latest:')
    st.write(underlying_new.index.max())


# smoothing_period = st.slider('Select a range of values', 1, 90, 25)
# st.write('Smoothing period: %d days' % smoothing_period)

# @st.cache
# def underlying_trend_figure(underlying, period):
#     fig = plt.figure(figsize=(23,14))
#     ax = fig.add_subplot(311)
#     ax.plot(underlying.Close.rolling(period).apply(lambda x: (x[-1]-x.mean())/x.std()))
#     ax.grid()
#     ax.set_title('Stationarized')
#     ax = fig.add_subplot(312)
#     ax.bar(underlying.index, underlying.Close.rolling(period).apply(lambda x: (x[-1]-x[0])/x[0]))
#     ax.set_title('Diff-ed')
#     ax.grid()
#     ax = fig.add_subplot(313)
#     ax.bar(underlying.index, underlying.Close.rolling(period).apply(lambda x: (x[-1]-x[0])).rolling(period).apply(lambda x: (x[-1]-x.mean())/x.std()))
#     ax.set_title('Diff-ed and Stationarized')
#     ax.grid()

#     return fig




# fig = underlying_trend_figure(underlying_df, smoothing_period)
# st.pyplot(fig)


# options_df = pd.concat([pd.read_csv(f, parse_dates=['Date'], index_col=['Date']) for f in glob.glob(DATA_PATH+'Nifty/Options/*.csv',)], 
#                     join='outer')


@st.cache
def calculate_period_delta(ohlc_df, period):
    period_change = ohlc_df.Close.rolling(period).apply(lambda x: (x[-1]-x[0])/x[0]).dropna()

    period_open = ohlc_df.Open.rolling(period).apply(lambda x: x[0]).dropna()

    period_high = ohlc_df.High.rolling(period).apply(lambda x: x.max()).dropna()

    period_low = ohlc_df.Low.rolling(period).apply(lambda x: x.min()).dropna()

    period_pos_change = (period_high - period_open)/period_open
    period_neg_change = (period_low - period_open)/period_open

    period_delta_df = pd.DataFrame({'delta':period_change, 'delta_max':period_pos_change, 'delta_min':period_neg_change}, )

    return period_delta_df

st.markdown('## Underlying Change Probability')   

trade_days_left = st.slider('Trade Days Left to Expiry', 1, 32, 5)
last_close = underlying_df['Close'][-1]
underlying_period_delta = calculate_period_delta(underlying_df, trade_days_left)
mean_delta = underlying_period_delta['delta'].mean()
std_delta = underlying_period_delta['delta'].std()

ax = sns.displot(data=underlying_period_delta['delta']*100,
                kde=True, 
                aspect=2.5)
ax.set_axis_labels(x_var='% Change in underlying')
plt.grid(axis='y')
plt.axvline(x=100*mean_delta, 
#             color='blue', 
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

st.latex(r"P(U > \overline{\Delta U} + \sigma)")
st.write("%.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_neg_sigma, underlying_sigma_minus, expected_change_neg_sigma*100))

expected_change_pos_sigma = mean_delta + std_delta
prob_change_pos_sigma = (underlying_period_delta['delta']>expected_change_pos_sigma).sum()/len(underlying_period_delta['delta'])
underlying_sigma_plus = last_close*(expected_change_pos_sigma+1)
st.write("%.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_pos_sigma, underlying_sigma_plus, expected_change_pos_sigma*100))

def round_to_base(x, base=50):
    return base * round(x/base)

with st.container() as cntnr:
    st.markdown()
    change_choice = st.radio('Price Or Change %', ('Price', 'Change'), index=0)
    if change_choice == 'Change':
        expected_changes = st.slider('Expected Change (%)', min_value=-10.0, max_value=10.0, value=(-1.5, 1.5), step=0.25)
        expected_high = expected_changes[1]/100
        expected_low = expected_changes[0]/100
    else:
        expected_changes = st.slider('Price Range', min_value=round_to_base(last_close*0.9), max_value=round_to_base(last_close*1.1, base=50), value=(round_to_base(last_close*1.015),round_to_base(last_close*0.985)), step=50)
        
        expected_high = (expected_changes[1]-last_close)/last_close
        expected_low = (expected_changes[0]-last_close)/last_close
        
    # Expiry change probability
    prob_change_plus = (underlying_period_delta['delta']>expected_high).sum()/len(underlying_period_delta['delta'])
    underlying_change_plus = last_close*(expected_high+1)

    prob_change_minus = (underlying_period_delta['delta']<expected_low).sum()/len(underlying_period_delta['delta'])
    underlying_change_minus = last_close*(expected_low+1)

    # Volatility change probability
    prob_change_within_range = ((underlying_period_delta['delta']>expected_low) & (underlying_period_delta['delta']<expected_high)).sum()/len(underlying_period_delta['delta'])
    prob_volatility_exceeds_range =  ((underlying_period_delta['delta_min']<expected_low) | (underlying_period_delta['delta_max']>expected_high)).sum()/len(underlying_period_delta['delta'])

    st.write("%.3f probability of expiry above %.2f (%.2f%% change)" %(prob_change_plus, underlying_change_plus, expected_high*100))
    st.write("%.3f probability of expiry below %.2f (%.2f%% change)" %(prob_change_minus, underlying_change_minus, expected_low*100))
    st.write("%.3f probability of expiry within %.2f and %.2f" %(prob_change_within_range, underlying_change_minus, underlying_change_plus))
    st.write("%.3f probability of price exceeding range at least once in %d trading days" %(prob_volatility_exceeds_range, trade_days_left))
    st.write("ie. %.2f %% chance of underlying reaching a price level above %.2f or below %.2f at least once before expiry" %(prob_volatility_exceeds_range*100, underlying_change_plus, underlying_change_minus))

st.header('Options')

options_path = os.path.join(underlying_path, 'Options')
options_df = load_ohlcv_files(options_path)

st.write('Earliest options data vailable:')
st.write(options_df.index.min())
st.write('Latest options data vailable:')
st.write(options_df.index.max())

with st.expander('Upload More Data') as option_upload_expander:

    underlying_upload_md = """Download Options chain data for indexes from here: 
                            https://www1.nseindia.com/products/content/derivatives/equities/historical_fo.htm
                            And upload it below to add it to your anylisis 
                            """
    uploaded_file = st.file_uploader("Upload More Data?", type="csv", help=underlying_upload_md, key='options_uploader')
    if uploaded_file is not None:
        options_new = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col=['Date'])
        st.write('New data uploaded for:')
        st.write('Earliest:')
        st.write(options_new.index.min())
        st.write('Closest:')
        st.write(options_new.index.max())
        st.write(options_new['Expiry'].unique().tolist())
        st.write(options_new['Option Type'].unique().tolist())



# Filter for latest data
options_df = options_df[options_df.index.max()==options_df.index]
expiry = st.selectbox('Choose Expiry', options_df['Expiry'].unique())
options_df = options_df[options_df['Expiry']==expiry]

st.write(options_df.head())
# dt.datetime.strptime(expiry, "%d-%b-%Y")




