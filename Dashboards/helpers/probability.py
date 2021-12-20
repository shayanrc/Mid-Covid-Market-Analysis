import streamlit as st
import pandas as pd


# @st.cache
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

@st.cache
def calculate_period_delta_cached(ohlc_df, period):
    return calculate_period_delta(ohlc_df, period)


def evaluate_pairs(position_pairs, underlying_close, delta_df, lot_size=50, expected_rate=0.05):
    
    required_returns = (position_pairs['cost']*(1+expected_rate))
    position_pairs['target'] = required_returns/lot_size
    
    # Change requrired to breakeven
    pe_breakeven_change = (position_pairs['pe_strike_price'] - position_pairs['target']) -underlying_close
    ce_breakeven_change = (position_pairs['ce_strike_price'] + position_pairs['target']) -underlying_close

    position_pairs['pe_breakeven_change'] = pe_breakeven_change
    position_pairs['ce_breakeven_change'] = ce_breakeven_change

    # Probabilty of breaking even due to volatility within the period
    pe_breakeven_probability = pe_breakeven_change.apply(lambda x:  (delta_df.delta_min<(x/underlying_close)).mean())
    ce_breakeven_probability = ce_breakeven_change.apply(lambda x:  (delta_df.delta_max>(x/underlying_close)).mean())
    position_pairs['pe_breakeven_probability'] = pe_breakeven_probability
    position_pairs['ce_breakeven_probability'] = ce_breakeven_probability
    position_pairs['pair_not_breakeven_probability'] = (1-pe_breakeven_probability)*(1-ce_breakeven_probability)
    position_pairs = position_pairs.sort_values('pair_not_breakeven_probability',  ignore_index=True)
    
    return position_pairs

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




