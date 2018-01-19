from fbprophet import Prophet

from keras import (
    callbacks,
    optimizers,
)
import keras.layers
from keras.layers import (
    Activation,
    Add,
    AlphaDropout,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D
)
from keras.models import Model
from keras.regularizers import l2

optimizer = optimizers.Adam(lr=0.1)
cbs = [
    callbacks.EarlyStopping(monitor='loss', patience=5000),
#    callbacks.ReduceLROnPlateau(monitor='loss', patience=10, factor=0.999),
]


def what_will_sell(num_ferments, months_before, dense_layers=3):
    inputs = Input(shape=(num_ferments*months_before+2,))
    x = inputs
    x = _fully_connected_layers(dense_layers, num_ferments, x)
    predictions = Dense(num_ferments, activation='sigmoid', use_bias=True)(x)
    return Model(inputs=inputs, outputs=predictions)


def prophet_integrated(num_ferments, months_before=1, dense_layers=3):
    last_sales = Input(shape=(num_ferments*months_before+2,))
    # Trend, yhat_lower, yhat_upper, seasonal, yhat
    prophet_outputs = Input(shape=(num_ferments, 5))
    x = Concatenate()([last_sales, Flatten()(prophet_outputs)])
    x = _fully_connected_layers(dense_layers, int(num_ferments * 1.5), x)
    predictions = Dense(num_ferments, activation='linear', use_bias=True)(x)
    return Model(inputs=[last_sales, prophet_outputs], outputs=predictions)


def create_model(num_ferments, months_before=1, dense_layers=3):
    last_sales = Input(shape=(num_ferments*months_before+2,))
    x = last_sales
    x = _fully_connected_layers(dense_layers, int(num_ferments * 1.5), x)
    predictions = Dense(num_ferments, activation='linear', use_bias=True)(x)
    return Model(inputs=last_sales, outputs=predictions)


def create_conv_model(num_ferments, months_before=1, dense_layers=3):
    last_orders = Input(shape=(num_ferments, months_before))
    x = last_orders
    for _ in range(2):
        for i in range(dense_layers):
            x = _default_dense_layer(num_ferments, x)
    last_orders_out = Flatten()(x)
    aux_data = Input(shape=(2, ))
    x = keras.layers.concatenate([last_orders_out, aux_data])
    x = _fully_connected_layers(dense_layers, num_ferments, x)
    predictions = Dense(num_ferments, activation='linear', use_bias=True)(x)
    return Model(inputs=[last_orders, aux_data], outputs=predictions)


def _default_dense_layer(num_neurons, x):
    x = Dense(
        num_neurons,
        kernel_regularizer=l2(0.02),
        kernel_initializer='lecun_normal',
        use_bias=True,
    )(x)
    return Activation('selu')(x)


def _dense_indentity(num_layers, num_neurons, x):
    x_shortcut = x
    for _ in range(num_layers):
        x = _default_dense_layer(num_neurons, x)
    x = Add()([x, x_shortcut])
    return Activation('selu')(x)


def _fully_connected_layers(num_layers, num_neurons, x):
    x = _default_dense_layer(num_neurons, x)
    for _ in range(num_layers // 4):
        x = _dense_indentity(4, num_neurons, x)
    x = _dense_indentity(max(num_layers % 4 - 1, 0), num_neurons, x)
    x = AlphaDropout(0.5)(x)
    return x
