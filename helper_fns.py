#helper_fns.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#23 January 2019
#
#helper functions used by x.py and various notebooks

#imports used below
import pandas as pd
import numpy as np

#read NYSE data into pandas dataframe
def read_nyse_data(data_path, start_date=None, end_date=None, drop_holidays=None):
    import glob
    files = glob.glob(data_path + '/*.txt')
    dataframes = []
    for file in files:
        df = pd.read_csv(file, parse_dates=['<date>'])
        dataframes.append(df)
    df = pd.concat(dataframes, ignore_index=True)
    df.columns = [col.replace('<', '').replace('>', '') for col in df.columns]
    df = df.sort_values(['date', 'ticker'])
    debug = False
    if (debug):
        print df.dtypes
        print 'number of records (M) = ', len(df)/1.0e6
        df.head()
    if (start_date):
        idx = (df['date'] >= start_date)
        df = df[idx]
    if (end_date):
        idx = (df['date'] <= end_date)
        df = df[idx]
    if (drop_holidays):
        daily_volume = df.groupby('date')['vol'].sum()
        idx = (daily_volume > 0)
        dates = daily_volume[idx].index
        idx = df['date'].isin(dates)
        df = df[idx]
    return df

#random buyers
def random_purchases(market_data, N_buyers, N_tickers, initial_dollars):
    random_buys = pd.DataFrame()
    dates = market_data['date'].unique()
    for buyer in range(N_buyers):
        rn_seed = 100 + buyer
        np.random.seed(seed=rn_seed)
        current_dollars = initial_dollars
        dates_dollars = {}
        for date in dates:
            idx = (market_data['date'] == date)
            market_1day = market_data[idx]
            all_tickers = market_1day['ticker'].unique()
            selected_tickers = np.random.choice(all_tickers, size=N_tickers, replace=False)
            idx = market_1day['ticker'].isin(selected_tickers)
            market_selected = market_1day[idx].sort_values('open', ascending=False)
            tickers = market_selected['ticker']
            open_price = market_selected['open']
            close_price = market_selected['close']
            dollars_per_ticker = current_dollars/N_tickers
            N_shares = (dollars_per_ticker/open_price).astype(int)
            total_share_cost = (N_shares*open_price).sum()
            cash = current_dollars - total_share_cost
            total_share_value = (N_shares*close_price).sum()
            current_dollars = cash + total_share_value
            dates_dollars[date] = current_dollars
        random_buys[buyer] = pd.Series(dates_dollars)
    random_buys = random_buys.sort_index()
    random_buys['dollars_mean'] = random_buys.mean(axis=1)
    random_buys['dollars_std'] = random_buys.std(axis=1)/np.sqrt(N_buyers - 1)
    return random_buys[['dollars_mean', 'dollars_std']]
