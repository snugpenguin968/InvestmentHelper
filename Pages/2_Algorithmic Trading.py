import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import numpy as np  # The Numpy numerical computing library
import pandas as pd  # The Pandas data science library
import requests  # The requests library for HTTP requests in Python
import xlsxwriter  # The XlsxWriter libarary for
import math  # The Python math module
from scipy import stats
from statistics import mean


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


IEX_CLOUD_API_TOKEN = 'Tpk_059b97af715d417d9f49f50b51b1c448'
stocks = pd.read_csv('sp_500_stocks.csv')
symbol = 'AAPL'
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data = requests.get(api_url).json()

st.set_page_config(page_title="Algorithmic Trading", layout='wide')
# st.markdown("<h1 style='margin-left:6em; '>Algorithmic Trading</h1>", unsafe_allow_html=True)
st.title('Algorithmic Trading')

col1, col2, col3 = st.columns(3)
with col1:
    st.write('Equal Weight S&P 500 Index Fund')
    st.caption('This will return an Excel file that will be saved to your device')
    with st.form('port_val'):
        portfolio_size = st.number_input('Enter the desired value of your portfolio ($): ', min_value=0)
        button = st.form_submit_button('Submit')



    if button:
        try:
            val = float(portfolio_size)

        except ValueError:
            st.write("That's not a valid number! \n Try again:")
            portfolio_size = input("Enter the value of your portfolio:")
        my_columns = ['Ticker', 'Price', 'Market Capitalization', 'Number Of Shares to Buy']
        final_dataframe = pd.DataFrame(columns=my_columns)
        final_dataframe = final_dataframe.append(
            pd.Series(['AAPL',
                       data['latestPrice'],
                       data['marketCap'],
                       'N/A'],
                      index=my_columns),
            ignore_index=True)
        symbol_groups = list(chunks(stocks['Ticker'], 100))
        symbol_strings = []
        for i in range(0, len(symbol_groups)):
            symbol_strings.append(','.join(symbol_groups[i]))
        #     print(symbol_strings[i])

        final_dataframe = pd.DataFrame(columns=my_columns)

        for symbol_string in symbol_strings:
            #     print(symbol_strings)
            batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
            data = requests.get(batch_api_call_url).json()
            for symbol in symbol_string.split(','):
                final_dataframe = final_dataframe.append(
                    pd.Series([symbol,
                               data[symbol]['quote']['latestPrice'],
                               data[symbol]['quote']['marketCap'],
                               'N/A'],
                              index=my_columns),
                    ignore_index=True)
        position_size = float(portfolio_size) / len(final_dataframe.index)
        for i in range(0, len(final_dataframe['Ticker']) - 1):
            final_dataframe.loc[i, 'Number Of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])
        final_dataframe.dropna(how='all')
        final_dataframe['Price'] = final_dataframe['Price'].apply(lambda x: f"${x:.2f}")

        for idx, row in final_dataframe.iterrows():
            if row['Market Capitalization'] is not None:
                final_dataframe.iloc[idx, 2] = "${:.2f}".format(row['Market Capitalization'])

        st.download_button('Get CSV', final_dataframe.to_csv(index=False), file_name='Recommended Trades.csv')

with col2:
    st.write('Quantitative Momentum Strategy')
    st.caption('This will return an Excel file that will be saved to your device')
    with st.form('port_val2'):
        portfolio_size = st.number_input('Enter the desired value of your portfolio ($): ', min_value=0)
        button = st.form_submit_button('Submit')

    if button:
        try:
            val = float(portfolio_size)
        except ValueError:
            print("That's not a number! \n Try again:")
            portfolio_size = input("Enter the value of your portfolio:")
        position_size = float(portfolio_size) / len(final_dataframe.index)
        symbol_groups = list(chunks(stocks['Ticker'], 100))
        symbol_strings = []
        for i in range(0, len(symbol_groups)):
            symbol_strings.append(','.join(symbol_groups[i]))

        my_columns = ['Ticker', 'Price', 'One-Year Price Return', 'Number of Shares to Buy']
        final_dataframe = pd.DataFrame(columns=my_columns)
        for symbol_string in symbol_strings:
            #     print(symbol_strings)
            batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
            data = requests.get(batch_api_call_url).json()
            for symbol in symbol_string.split(','):
                final_dataframe = final_dataframe.append(
                    pd.Series([symbol,
                               data[symbol]['quote']['latestPrice'],
                               data[symbol]['stats']['year1ChangePercent'],
                               'N/A'
                               ],
                              index=my_columns),
                    ignore_index=True)

        final_dataframe.sort_values('One-Year Price Return', ascending=False, inplace=True)
        final_dataframe = final_dataframe[:51]
        final_dataframe.reset_index(drop=True, inplace=True)
        hqm_columns = [
            'Ticker',
            'Price',
            'Number of Shares to Buy',
            'One-Year Price Return',
            'One-Year Return Percentile',
            'Six-Month Price Return',
            'Six-Month Return Percentile',
            'Three-Month Price Return',
            'Three-Month Return Percentile',
            'One-Month Price Return',
            'One-Month Return Percentile',
            'HQM Score'
        ]

        hqm_dataframe = pd.DataFrame(columns=hqm_columns)

        for symbol_string in symbol_strings:
            #     print(symbol_strings)
            batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
            data = requests.get(batch_api_call_url).json()
            for symbol in symbol_string.split(','):
                hqm_dataframe = hqm_dataframe.append(
                    pd.Series([symbol,
                               data[symbol]['quote']['latestPrice'],
                               'N/A',
                               data[symbol]['stats']['year1ChangePercent'],
                               'N/A',
                               data[symbol]['stats']['month6ChangePercent'],
                               'N/A',
                               data[symbol]['stats']['month3ChangePercent'],
                               'N/A',
                               data[symbol]['stats']['month1ChangePercent'],
                               'N/A',
                               'N/A'
                               ],
                              index=hqm_columns),
                    ignore_index=True)
        hqm_dataframe.dropna(inplace=True)
        time_periods = [
            'One-Year',
            'Six-Month',
            'Three-Month',
            'One-Month'
        ]

        for row in hqm_dataframe.index:
            for time_period in time_periods:
                hqm_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(
                    hqm_dataframe[f'{time_period} Price Return'],
                    hqm_dataframe.loc[row, f'{time_period} Price Return']) / 100

        for row in hqm_dataframe.index:
            momentum_percentiles = []
            for time_period in time_periods:
                momentum_percentiles.append(hqm_dataframe.loc[row, f'{time_period} Return Percentile'])
            hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)

        # hqm_dataframe.sort_values(by = 'HQM Score', ascending = False,inplace=True)
        hqm_dataframe = hqm_dataframe[:51]
        position_size = float(portfolio_size) / len(hqm_dataframe.index)
        for i in range(0, len(hqm_dataframe['Ticker']) - 1):
            hqm_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])
        hqm_dataframe.sort_values(by='HQM Score', ascending=False, inplace=True)
        hqm_dataframe['Price'] = hqm_dataframe['Price'].apply(lambda x: f"${x:.2f}")
        for time_period in time_periods:
            hqm_dataframe[f'{time_period} Return Percentile'] = hqm_dataframe[f'{time_period} Return Percentile'].apply(
                lambda x: f"{x * 100:.2f}%")
            hqm_dataframe[f'{time_period} Price Return'] = hqm_dataframe[f'{time_period} Price Return'].apply(
                lambda x: f"{x * 100:.2f}%")
        hqm_dataframe['HQM Score'] = hqm_dataframe['HQM Score'].apply(lambda x: f"{x * 100:.2f}%")
        st.download_button('Get CSV', hqm_dataframe.to_csv(index=False), file_name='momentum_strategy.csv')

with col3:
    st.write('Quantitative Value Strategy')
    st.caption('This will return an Excel file that will be saved to your device')
    with st.form('port_val3'):
        portfolio_size = st.number_input('Enter the desired value of your portfolio ($): ', min_value=0)
        button = st.form_submit_button('Submit')
    if button:
        try:
            val = float(portfolio_size)
        except ValueError:
            print("That's not a number! \n Try again:")
            portfolio_size = input("Enter the value of your portfolio:")
        symbol_groups = list(chunks(stocks['Ticker'], 100))
        symbol_strings = []
        for i in range(0, len(symbol_groups)):
            symbol_strings.append(','.join(symbol_groups[i]))
        my_columns = ['Ticker', 'Price', 'Price-to-Earnings Ratio', 'Number of Shares to Buy']
        final_dataframe = pd.DataFrame(columns=my_columns)
        for symbol_string in symbol_strings:
            #     print(symbol_strings)
            batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
            data = requests.get(batch_api_call_url).json()
            for symbol in symbol_string.split(','):
                final_dataframe = final_dataframe.append(
                    pd.Series([symbol,
                               data[symbol]['quote']['latestPrice'],
                               data[symbol]['quote']['peRatio'],
                               'N/A'
                               ],
                              index=my_columns),
                    ignore_index=True)
        final_dataframe.sort_values('Price-to-Earnings Ratio', inplace=True)
        final_dataframe = final_dataframe[final_dataframe['Price-to-Earnings Ratio'] > 0]
        final_dataframe = final_dataframe[:50]
        final_dataframe.reset_index(inplace=True)
        final_dataframe.drop('index', axis=1, inplace=True)
        position_size = float(portfolio_size) / len(final_dataframe.index)
        for i in range(0, len(final_dataframe['Ticker'])):
            final_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])
        symbol = 'AAPL'
        batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=advanced-stats,quote&symbols={symbol}&token={IEX_CLOUD_API_TOKEN}'
        data = requests.get(batch_api_call_url).json()

        # P/E Ratio
        pe_ratio = data[symbol]['quote']['peRatio']

        # P/B Ratio
        pb_ratio = data[symbol]['advanced-stats']['priceToBook']

        # P/S Ratio
        ps_ratio = data[symbol]['advanced-stats']['priceToSales']

        # EV/EBITDA
        enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
        ebitda = data[symbol]['advanced-stats']['EBITDA']
        ev_to_ebitda = enterprise_value / ebitda

        # EV/GP
        gross_profit = data[symbol]['advanced-stats']['grossProfit']
        ev_to_gross_profit = enterprise_value / gross_profit
        rv_columns = [
            'Ticker',
            'Price',
            'Number of Shares to Buy',
            'Price-to-Earnings Ratio',
            'PE Percentile',
            'Price-to-Book Ratio',
            'PB Percentile',
            'Price-to-Sales Ratio',
            'PS Percentile',
            'EV/EBITDA',
            'EV/EBITDA Percentile',
            'EV/GP',
            'EV/GP Percentile',
            'RV Score'
        ]

        rv_dataframe = pd.DataFrame(columns=rv_columns)

        for symbol_string in symbol_strings:
            batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
            data = requests.get(batch_api_call_url).json()
            for symbol in symbol_string.split(','):
                enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
                ebitda = data[symbol]['advanced-stats']['EBITDA']
                gross_profit = data[symbol]['advanced-stats']['grossProfit']

                try:
                    ev_to_ebitda = enterprise_value / ebitda
                except TypeError:
                    ev_to_ebitda = np.NaN

                try:
                    ev_to_gross_profit = enterprise_value / gross_profit
                except TypeError:
                    ev_to_gross_profit = np.NaN

                rv_dataframe = rv_dataframe.append(
                    pd.Series([
                        symbol,
                        data[symbol]['quote']['latestPrice'],
                        'N/A',
                        data[symbol]['quote']['peRatio'],
                        'N/A',
                        data[symbol]['advanced-stats']['priceToBook'],
                        'N/A',
                        data[symbol]['advanced-stats']['priceToSales'],
                        'N/A',
                        ev_to_ebitda,
                        'N/A',
                        ev_to_gross_profit,
                        'N/A',
                        'N/A'
                    ],
                        index=rv_columns),
                    ignore_index=True
                )
        for column in ['Price-to-Earnings Ratio', 'Price-to-Book Ratio', 'Price-to-Sales Ratio', 'EV/EBITDA', 'EV/GP']:
            rv_dataframe[column].fillna(rv_dataframe[column].mean(), inplace=True)
        metrics = {
            'Price-to-Earnings Ratio': 'PE Percentile',
            'Price-to-Book Ratio': 'PB Percentile',
            'Price-to-Sales Ratio': 'PS Percentile',
            'EV/EBITDA': 'EV/EBITDA Percentile',
            'EV/GP': 'EV/GP Percentile'
        }

        for row in rv_dataframe.index:
            for metric in metrics.keys():
                rv_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(rv_dataframe[metric],
                                                                                 rv_dataframe.loc[row, metric]) / 100

        for row in rv_dataframe.index:
            value_percentiles = []
            for metric in metrics.keys():
                value_percentiles.append(rv_dataframe.loc[row, metrics[metric]])
            rv_dataframe.loc[row, 'RV Score'] = mean(value_percentiles)
        rv_dataframe.sort_values(by = 'RV Score', inplace = True)
        rv_dataframe = rv_dataframe[:50]
        rv_dataframe.reset_index(drop = True, inplace = True)
        position_size = float(portfolio_size) / len(rv_dataframe.index)
        for i in range(0, len(rv_dataframe['Ticker'])-1):
            rv_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / rv_dataframe['Price'][i])
        rv_dataframe.loc[:,'Price']='$' + rv_dataframe.loc[:, 'Price'].astype(float).round(2).astype(str)
        for column in ['Price-to-Earnings Ratio','Price-to-Book Ratio','Price-to-Sales Ratio','EV/EBITDA','EV/GP']:
            rv_dataframe[column]=rv_dataframe[column].apply(round)
        for column in ['PE Percentile','PB Percentile','PS Percentile','EV/EBITDA Percentile','EV/GP Percentile','RV Score']:
            rv_dataframe[column]=rv_dataframe[column].apply(lambda x: f"{x*100:.1f}%")
        st.download_button('Get CSV', rv_dataframe.to_csv(index=False), file_name='value_strategy.csv')



