import streamlit as st
import pandas as pd
import pydeck as pdk
from urllib.error import URLError
import datetime as dt
from datetime import date
import copy
import streamlit.components.v1 as components
import yahoo_fin.stock_info as si
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from pandas_datareader import data as web # Reads stock data
import matplotlib.pyplot as plt # Plotting
import matplotlib.dates as mdates # Styling dates
from pandas.tseries.offsets import BDay
import datetime as dt # For defining dates
import mplfinance as mpf # Matplotlib finance
from datetime import datetime
import time
import copy
# Used to get data from a directory
import os
from os import listdir
from os.path import isfile, join
#Statsmodels is a great library we can use to run regressions.
import statsmodels.api as sm
# Seaborn extends the capabilities of Matplotlib
import seaborn as sns
import yfinance as yf
# Used for calculating regressions
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
plt.style.use('fivethirtyeight')
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices


def get_from_yahoo(ticker, start,end):
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.index=df.index.date
    
    return df
def add_daily_return_to_df(df):
    df['daily_return']=(df['Adj Close']/df['Adj Close'].shift(1))-1
    return df
def merge_df_by_column_name(col_name, start,end, tickers):
    mult_df = pd.DataFrame()
    for x in tickers:
        df = get_from_yahoo(x,start,end)
        mask = (df.index >= start) & (df.index <= end)
        mult_df[x] = df.loc[mask][col_name]
    return mult_df
def find_port_beta(port_df, start,end):
    # Will hold data for S&P and my portfolio
    mult_df = pd.DataFrame()
    # Mask defining the dates worth of data that we want
    port_mask = (port_df.index >= start) & (port_df.index <= end)

    # Get S&P Dataframe
    sp_df = get_from_yahoo('^GSPC',start,end)
    sp_df=add_daily_return_to_df(sp_df)
    sp_mask = (sp_df.index >= start) & (sp_df.index <= end)

    # Add S&P daily return to dataframe
    mult_df['^GSPC'] = sp_df.loc[sp_mask]['daily_return']

    # Add the portfolio daily return data
    mult_df['Portfolio'] = port_df.loc[port_mask]['daily_return']
    #mult_df['Portfolio'] = port_df.loc[port_mask]
    # Provides the covariance between the securities
    cov = mult_df.cov() * 252

    # Get the covariance of the stocks and the market
    cov_vs_market = cov.iloc[0,1]

    # Get annualized variance of the S&P
    sp_var = mult_df['^GSPC'].var() * 252

    # Beta is normally calculated over a 5 year period which is why you may see a difference
    beta = cov_vs_market / sp_var
    return beta
def get_prices_on_date(stocks_df, date):
    return stocks_df.loc[pd.DatetimeIndex([date])]['Adj Close'].item()
def get_port_val_by_date(start, shares, tickers):
    port_prices = merge_df_by_column_name('Adj Close',  start,start, tickers)
    port_prices = port_prices.values.tolist()
    port_prices = sum(port_prices, [])
    value_list = []
    for price, share in zip(port_prices, shares):
        value_list.append(price * share)
    return sum(value_list)
def add_port_val_col(df,shares):
    li=[]
    for index,row in df.iterrows():
        row_prices=df.loc[index].tolist()
        tot=0
        for price,share in zip(row_prices,shares):
            tot+=price*share
        li.append(tot)
    df['Value']=li
    return df
def stock_values(df,shares):
    temp=[]
    i=0
    for column in df.columns.tolist()[:-1]:
        temp=df[column]*shares[i]
        i+=1
        df[column]=temp

    return df
def get_port_daily_return(start,end, shares, tickers):
    mult_df = merge_df_by_column_name('Adj Close',  start,end, tickers)
    num_cols = len(mult_df.columns)
    i = 0
    while i < num_cols:
        mult_df[tickers[i]] = mult_df[tickers[i]].apply(lambda x: x * shares[i])
        i += 1
    mult_df['Total'] = mult_df.iloc[:, 0:num_cols].sum(axis=1)
    mult_df['daily_return'] = (mult_df['Total'] / mult_df['Total'].shift(1)) - 1
    return mult_df 
def value_chart(mult_df):
    

    chart_ptfvalue = go.Figure()  # generating a figure that will be updated in the following lines
    chart_ptfvalue.add_trace(go.Scatter(x=mult_df.index, y=mult_df.Value,
                                        mode='lines',  # you can also use "lines+markers", or just "markers"
                                        name='Global Value'))
   
    chart_ptfvalue.layout.height=500
    chart_ptfvalue.update_layout(margin = dict(t=0, b=0, l=0, r=0))  # this will help you optimize the chart space
    chart_ptfvalue.update_layout(title='Total Value of Portfolio Over Time',title_x=0.52,margin = dict(t=50, b=50, l=0, r=0))
    chart_ptfvalue.update_layout(
        #     title='Global Portfolio Value (USD $)',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Value: $ USD',
            titlefont_size=14,
            tickfont_size=12,
        ))
    
    return chart_ptfvalue

def donut(all_stock_values):
    means=np.round(np.mean(all_stock_values.drop('Value',axis=1),axis=0),2)
    donut_top = go.Figure()
    donut_top.add_trace(go.Pie(labels=all_stock_values.columns.tolist(), values=means))
    donut_top.update_traces(hole=.7, hoverinfo="label+value+percent")
    donut_top.update_traces(textposition='outside', textinfo='label+value')
    donut_top.update_layout(showlegend=False)
    donut_top.update_layout(title='Top Holdings ($)',title_x=0.52,margin = dict(t=50, b=50, l=0, r=0))
    return donut_top

def daily_returns(all_stock_values):
    
    fig2 = go.Figure(data=[
        go.Bar(name='Portfolio', x=mult_df.index, y=(all_stock_values['Value'].pct_change()*100).round(2)),
        go.Bar(name='SP500', x=sp_df.index, y=(sp_df['Adj Close'].pct_change()*100).round(2))
    ])
    # Change the bar mode
    fig2.update_layout(barmode='group')
    
    fig2.layout.height=300
    fig2.update_layout(title='Daily Return (%)',title_x=0.50,margin = dict(t=50, b=50, l=0, r=0))
    fig2.update_layout(
        #     title='% variation - Portfolio vs SP500',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='% change',
            titlefont_size=14,
            tickfont_size=12,
        ))
    fig2.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99))

    return fig2


st.set_page_config(page_title="Portfolio Visualizer", page_icon="📁")
st.markdown("<h1 style='text-align: center; '>Portfolio Visualizer</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Build your portfolio here</h3>", unsafe_allow_html=True)
dates_input=st.date_input('This site supports stocks ranging from January 1, 2000 to the current date.',value=(dt.date(2020,1,1),date.today()),max_value=date.today(),min_value=dt.date(2000,1,1))
stock_list=pd.read_csv('Wilshire-5000-Stocks.csv')['Ticker'].tolist()
stock_list.append('META') 
options=st.multiselect('Select stock tickers (Wilshire 5000)',stock_list)


    

shares_dict={option:0 for option in options}
with st.form('shares'):
        st.markdown("<h5 style='text-align: center;'>Enter the Number of Shares for Each Stock</h5>", unsafe_allow_html=True)  
        for k,v in shares_dict.items():   
            shares_dict[k]=st.number_input(k,min_value=0)
        val=st.number_input('Enter amount of money you are willing to put into the portfolio ($)',min_value=1)
        columns = st.columns((2, 1, 2))
        shares_submitted=columns[1].form_submit_button('Submit')

        st.caption('The search query may take a while to run depending on the size of your portfolio.')
if shares_submitted:
    sp_df = web.DataReader('^GSPC', 'yahoo', dates_input[0],dates_input[1])
    mult_df=(merge_df_by_column_name('Adj Close',dates_input[0],dates_input[1],options)) 
    shares=[i for i in shares_dict.values()]
    mult_df=add_port_val_col(mult_df,shares)
    all_stock_values=stock_values(copy.deepcopy(mult_df),shares) 
    tot_port_df = get_port_daily_return(dates_input[0],dates_input[1],shares, options)
    st.markdown("<h3 style='text-align: center;'>Visualizations</h3>", unsafe_allow_html=True)
    st.plotly_chart(value_chart(all_stock_values))
    st.plotly_chart(daily_returns(all_stock_values))
    st.plotly_chart(donut(all_stock_values))
    st.markdown("<h3 style='text-align: center;'>Key Statistics</h3>", unsafe_allow_html=True)
    port_val_start = tot_port_df.iloc[0,-2]
    #st.markdown("<p style='text-align: center;'>Portfolio Value at Start of Timeframe : $%2.2f</p>" % (port_val_start),unsafe_allow_html=True)
    st.write(f'Portfolio Value at Start of Timeframe : ${(port_val_start*100).round(2)}')
    port_val_end = tot_port_df.iloc[-1,-2]
    #st.markdown("<p style='text-align: center;'>Portfolio Value at End of Timeframe : $%2.2f </p>" % (port_val_end),unsafe_allow_html=True)
    st.write(f'Portfolio Value at End of Timeframe : ${(port_val_end*100).round(2)}')
    roi_port = (port_val_end - port_val_start) / port_val_end
    #st.markdown("<p style='text-align: center;'>Portfolio ROI at End of Timeframe : %2.2f %% </p>" % (roi_port * 100),unsafe_allow_html=True)
    st.write(f'Portfolio ROI at End of Timeframe : {(roi_port*100).round(2)}%')
    # S&P ROI
    sp_df=add_daily_return_to_df(sp_df)
    sp_val_start = sp_df.iloc[0,-2]
    sp_val_end = sp_df.iloc[-1,-2]
    sp_roi = (sp_val_end - sp_val_start) / sp_val_end
    #st.markdown("<p style='text-align: center;'>S&P ROI at End of Timeframe : %2.2f %%</p>" % (sp_roi * 100),unsafe_allow_html=True)
    st.write(f'S&P ROI at End of Timeframe : {(sp_roi*100).round(2)}%')
    risk_free_rate=0.013
    port_beta = find_port_beta(tot_port_df, dates_input[0],dates_input[1])
    #st.markdown("<p style='text-align: center;'>Portfolio Beta : %2.2f %%</p>" % (port_beta * 100),unsafe_allow_html=True)
    st.write(f'Portfolio Beta : {(port_beta*100).round(2)}%')
    port_alpha = roi_port - risk_free_rate - (port_beta * (sp_roi - risk_free_rate))
    #st.markdown("<p style='text-align: center;'>Portfolio Alpha : %2.2f %%</p>" % (port_alpha * 100),unsafe_allow_html=True )
    st.write(f'Portfolio Alpha : {(port_alpha*100).round(2)}%')
    col1,col2=st.columns(2)
    returns=mult_df.pct_change().drop('Value',axis=1)
    with col1:
        st.caption('Correlation Matrix')
        st.write(mult_df.drop('Value',axis=1).corr())
    with col2:
        st.caption('Annualized Covariance Matrix')
        st.write(returns.cov()*252)
    st.markdown("<h3 style='text-align: center;'>Optimization</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>*Optimized for the max Sharpe Ratio</h6>", unsafe_allow_html=True)
        #st.caption('Optimized for the max Sharpe Ratio')
    mu=expected_returns.mean_historical_return(mult_df)
    S=risk_models.sample_cov(mult_df)
    try:
        #optimize for max Sharpe Ratio
        ef=EfficientFrontier(mu,S)
        weights=ef.max_sharpe()
        cleaned_weights=ef.clean_weights()
        performance=(ef.portfolio_performance(verbose=True))
        #st.markdown(f"<p style='text-align: center;'>Expected Annual Return : {(performance[0]*100).round(2)}%</p>",unsafe_allow_html=True )
        st.write(f'Expected Annual Return : {(performance[0]*100).round(2)}%')
        #st.markdown(f"<p style='text-align: center;'>Annual Volatility : {(performance[1]*100).round(2)}%</p>",unsafe_allow_html=True )
        st.write(f'Annual Volatility : {(performance[1]*100).round(2)}%')
        #st.markdown(f"<p style='text-align: center;'>Sharpe Ratio : {(performance[2]).round(2)}</p>",unsafe_allow_html=True )
        st.write(f'Sharpe Ratio : {(performance[2]).round(2)}')
        #st.markdown("<h6 style='text-align: center;'>This segment will return a suggested Discrete Allocation (number of shares you should buy for each stock)</h6>", unsafe_allow_html=True)
        st.caption('This segment will return a suggested Discrete Allocation (number of shares you should buy for each stock)')
        latest_prices=get_latest_prices(mult_df)
        weights=cleaned_weights
        da=DiscreteAllocation(weights,latest_prices,total_portfolio_value=val) # how much money you are willing to put into this portfolio
        allocation,leftover=da.lp_portfolio()
        for a,v in allocation.items():
            if 'Value' in allocation.keys():
                st.write(f'{options[0]: v}')
            else:
                st.write(f'{a} : {v}')
            #st.markdown(f"<p style='text-align: center;'>{a} : {v}</p>",unsafe_allow_html=True )
        #st.markdown(f"<p style='text-align: center;'>Funds Remaining After Suggested Allocation: ${leftover.round(2)}</p>",unsafe_allow_html=True )
        st.write(f'Funds Remaining After Suggested Allocation: ${leftover.round(2)}')
    except Exception as e:
        st.markdown(f"<p style='text-align: center;'>This combination of stocks, shares, and funds is unable to be optimized.</p>",unsafe_allow_html=True )
           
                




        
    

    
    

