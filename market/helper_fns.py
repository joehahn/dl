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
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    return df

#resample market timeseries data...delta=fractional change in open across each time-bin,
#and std=std(open)/open
def resample_data(df, freq, tickers=None):
    if (tickers):
        idx = df['ticker'].isin(tickers)
        dfc = df[idx].copy()
    else:
        dfc = df.copy()
    dfc['mean'] = dfc['open']
    dfc['std'] = dfc['open']
    dfc['Ndates'] = dfc['date']
    dfc['Ntickers'] = 0
    dfc['day'] = dfc['date'].dt.dayofweek
    dfc['$vol'] = dfc['open']*dfc['vol']
    aggregator = {'date':'first', 'Ndates':'count', 'day':'first', 'open':'first', 'close':'last', 
        'std':'std', 'vol':'sum', '$vol':'sum'}
    groups = [pd.Grouper(freq=freq, key='date'), 'ticker']
    dfr = dfc.groupby(groups, as_index=False).agg(aggregator)
    dfr['delta'] = (dfr['close'] - dfr['open'])/dfr['open']
    dfr['std'] /= dfr['open']
    dfr['G$vol'] = dfr['$vol']/1.0e9
    ###df_resampled['date_lagged'] = df_resampled.groupby(['ticker'])['date'].shift(1)
    cols = ['date', 'Ndates', 'day', 'ticker', 'vol', 'G$vol', 'open', 'close', 'delta', 'std']
    dfc = dfr[cols]
    #pivot table
    cols = ['date', 'ticker', 'close', 'G$vol', 'std', 'delta']
    dfp = dfc[cols].pivot(index='date', columns='ticker')
    #lag feature columns
    lag = 1
    cols = ['close', 'G$vol', 'std']
    for col in cols:
        dfp[col] = dfp[col].shift(periods=lag, axis=0)
    #drop the multi-index
    cols = [col[0]+'_'+col[1] for col in dfp.columns.tolist()]
    dfp.columns = cols
    #drop first null row
    dfp = dfp[1:]
    #drop rows at least 10% null...happens when a new stock starts trading midweek
    keepers = pd.Series()
    for idx, row in dfp.iterrows():
        keepers[idx] = True
        if (row.isnull().sum() > 0.1*len(row)):
            keepers[idx] = False
    dfk = dfp[keepers]
    #drop columns containing 1+ nulls...happens when new stock starts trading, trading is halted etc
    #...need to handle this better
    dfk = dfk.dropna(axis=1, how='any')
    return dfk, dfr

#this helper function builds a simple MLP classifier
def mlp_classifier(layers, dropout_fraction=None):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    model = Sequential()
    N_inputs = layers[0]
    model.add(Dense(N_inputs, activation='elu', input_shape=(N_inputs,)))
    if (dropout_fraction):
        model.add(Dropout(dropout_fraction))
    for N in layers[1:-1]:
        model.add(Dense(N, activation='elu'))
        if (dropout_fraction):
            model.add(Dropout(dropout_fraction))
    N_outputs = layers[-1]
    model.add(Dense(N_outputs, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
