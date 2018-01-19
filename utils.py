import csv

from fbprophet import Prophet
import keras.backend as K
from keras.utils import normalize
import numpy as np
import pandas as pd


def construct_input(output, months_before=1):
    num_months = output.shape[0]
    num_items = output.shape[1]
    input_ = np.zeros((num_months + months_before, num_items*months_before + 2))
    for month, orders in enumerate(output):
        for mb in range(months_before):
            input_[month + mb + 1, mb*num_items : (mb+1)*num_items] = orders
        year = month // 12
        month_in_year = month % 12
        input_[month + months_before, -2:] = (year, month)
    return input_


def construct_output(data):
    """Transforms month-to-month sales records into matrix."""
    ferments = []
    years = []
    num_months = 12
    ferments = sorted(list(set(data[:, 0])))
    tf = dict([(t, i) for i, t in enumerate(ferments)])
    years = sorted(list(set(data[:, -2])))
    ty = dict([(t, i) for i, t in enumerate(years)])
    output = np.zeros((len(years) * num_months, len(ferments)))
    for record in data:
        output[
                ty[record[-2]]*num_months + int(record[-1]) - 1,
                tf[record[0]]
        ] = float(record[3])
    mask = output == 0  #We use mask to not run into stack overflow
    output = output[np.flatnonzero((~mask).sum(axis=1)), :]
    return output, ferments


def construct_prophet_outputs(
        ferment_dataframes,
        val_split=0,
        future_periods=0
    ):
    if val_split:
        for df in ferment_dataframes:
            df = df[-int(val_split*len(df['ds'])):]
    m = Prophet()
    m.fit(ferment_dataframes[0])
    future = m.make_future_dataframe(periods=future_periods, freq='M')
    prophet_outputs = np.zeros((
        len(ferment_dataframes[0]['ds']) + future_periods,
        len(ferment_dataframes),
        5
    ))
    for i, ferment_dataframe in enumerate(ferment_dataframes):
        m = Prophet(seasonality_prior_scale=1)
        m.fit(ferment_dataframe)
        fcst = m.predict(future)
        # Trend, yhat_lower, yhat_upper, seasonal, yhat
        interesting_fcst = {
            'trend': fcst['trend'],
            'yhat_lower': fcst['yhat_lower'],
            'yhat_upper': fcst['yhat_upper'],
            'seasonal': fcst['seasonal'],
            'yhat': fcst['yhat'],
        }
        prophet_outputs[:, i, :] = pd.DataFrame(data=interesting_fcst)
    return prophet_outputs


def construct_data_for_model(
        data,
        months_before=12,
        val_split=0.1,
        val_mask=None):
    def split_data(data):
        return data[~val_mask], data[val_mask]

    output, ferments = construct_output(data)
    output = np.log(output[:-1] + 1e-7)
    feed = normalize(
        construct_input(output, months_before)[months_before:-months_before])
    dates = np.array(
        [[t // 12, t % 12] for t in range(len(output))][months_before:])
    usable_output = output[months_before:]
    sold_at_all = np.cast['bool'](usable_output)
    if not val_mask:
        val_mask = np.zeros((len(usable_output)), dtype='bool')
        val_mask[-int(len(val_mask)*val_split):] = True
    return (
        output, ferments,
        *split_data(feed),
        *split_data(dates),
        *split_data(usable_output),
        *split_data(sold_at_all),
        val_mask,
    )


def purchase_mean_absolute_error(y_true, y_pred):
    """Negative predictions are set to 0 before calculating error."""
    ma = K.cast(y_pred > 0, dtype='float32')
    return K.mean(K.abs(y_true - ma*y_pred))

def purchase_mean_error_over_mean(y_true, y_pred):
    return purchase_mean_absolute_error(y_true, y_pred)/K.mean(y_true)
