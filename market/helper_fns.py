#helper_fns.py
#
#by Joe Hahn
#jmh.datasciences@gmail.com
#23 January 2019
#
#helper functions used by x.py and various notebooks

#imports used below
import pandas as pd
#import numpy as np

#read zipped market data into pandas dataframe
def read_market_data(path, start_date=None, end_date=None, drop_holidays=None, debug=False):
    import glob
    files = glob.glob(path)
    files.sort()
    from zipfile import ZipFile
    df = pd.DataFrame()
    for file in files:
        print 'reading file = ', file
        zfile = ZipFile(file)
        df_list = [pd.read_csv(zfile.open(one_file.filename), parse_dates=['<date>']) \
            for one_file in zfile.infolist()]
        for one_df in df_list:
           one_df.columns = [col.strip('<').strip('>') for col in one_df.columns]
           df = df.append(one_df, ignore_index=True) 
    if (debug):
        print df.dtypes
        print 'number of records (M) = ', len(df)/1.0e6
    if (start_date):
        idx = (df['date'] >= start_date)
        df = df[idx]
    if (end_date):
        idx = (df['date'] <= end_date)
        df = df[idx]
    if (drop_holidays):
        daily_volume = df.groupby('date')['vol'].sum()
        idx = (daily_volume > 0)
        non_holidays = daily_volume[idx].index
        idx = df['date'].isin(non_holidays)
        df = df[idx]
    df = df.sort_values(['date', 'ticker'])
    return df

#resample market timeseries data
def resample_data(df, freq):
    df['std'] = 0.5*(df['open'] + df['close'])
    aggregator = {'date':'first', 'open':'first', 'close':'last', 'std':'std', 'vol':'sum'}
    groups = [pd.Grouper(freq=freq, key='date'), 'ticker']
    cols = ['date', 'ticker', 'open', 'std', 'close', 'vol']
    df_resampled = df.groupby(groups, as_index=False).agg(aggregator)[cols]
    df_resampled['date_lagged'] = df_resampled.groupby(['ticker'])['date'].shift(1)
    cols = ['date', 'date_lagged', 'ticker', 'open', 'std', 'close', 'vol']
    dfd = df_resampled.dropna()[cols]
    return dfd
