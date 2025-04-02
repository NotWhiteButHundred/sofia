from datetime import timedelta
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data(query):
    return pd.read_csv(os.path.dirname(__file__)
        + f'/{query}.csv')


def load_as_simple_df(query, facets_str, sampling_rate, values=None, start_datetime=None, end_datetime=None):
    
    print('query= {}; sampling_rate= {}s'.format(query, sampling_rate))

    # Load original data
    data = load_data(query)
    data['datetime'] = pd.to_datetime(data['datetime'])

    if start_datetime is not None:
        data = data[lambda x: x['datetime'] >= pd.to_datetime(start_datetime)]
    if end_datetime is not None:
        data = data[lambda x: x['datetime'] < pd.to_datetime(end_datetime)]

    min_datetime = data['datetime'].min()
    max_datetime = data['datetime'].max()
    print("Duration:", min_datetime, max_datetime)

    # Simplify time-mode
    data['datetime'] -= min_datetime
    data['datetime'] //= timedelta(seconds=sampling_rate)

    # Simplify non-time-mode
    facets = facets_str.split('/')
    oe = preprocessing.OrdinalEncoder()
    data[facets] = oe.fit_transform(data[facets]).astype(int)

    if values == None:
        data = data[['datetime'] + facets]
        data['value'] = 1.
    else:
        data = data[['datetime'] + facets + [values]]
        data[values] = data[values].astype(np.float64)

    data = data.groupby(['datetime'] + facets).sum().reset_index()

    return data, min_datetime, max_datetime, [{category: idx for idx, category in enumerate(categories)} for categories in oe.categories_]