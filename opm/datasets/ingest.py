from typing import List
import datetime
import tqdm
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf

def fetch_historical_data_from_yfinance(security_names: List[str],
                              start_date: datetime.date,
                              end_date: datetime.date,
                                        verbose: bool = True) ->  pd.DataFrame:
    """Fetches historical daily prices/volumes of a list of securities from yahoo finance """
    historical_data = []
    for name in security_names:
        if verbose:
            print(f'Downloading {name}')
        df = yf.download(name, start=start_date, end=end_date)
        df['Security']    = name
        df['Date']   = df.index.tolist()
        historical_data.append(df)

    dataset = pd.concat(historical_data)
    dataset.reset_index(drop = True, inplace= True)
    return dataset

