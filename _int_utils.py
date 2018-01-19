import csv
from importlib import reload

import numpy as np
import pandas as pd

import model
import utils

def read_datafile(filepath='cleaned_KIEKINIAI_WERK_BEECH_working_copy.csv'):
    data = []
    with open(filepath, 'r') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            data.append(row)
        return np.array(data)


def read_data_for_prophet(filepath='with-zeros.csv'):
    # Drop last month info, which may be incomplete
    df = pd.read_csv(filepath, parse_dates=[0])[:-1]
    dfs = [pd.DataFrame(data={'ds': df['DS'], 'y': df[k]}) for k in df][1:]
    return dfs
