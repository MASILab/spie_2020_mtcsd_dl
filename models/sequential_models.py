from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Dropout, merge, concatenate, Convolution3D, Flatten
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Add
from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, adam, nadam, Adagrad, RMSprop
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, CSVLogger

from utils.losses import fracvol_loss
from utils.metrics import calc_acc, frac_loss, sh_loss

def build_nn_resnet():
    # Batch Size 1000 and 250 epochs, hits a median of 0.8125
    input_dims = 45
    inputs = Input(shape=(input_dims,))

    #split0 = Input(shape=(1,))
    #split1 = Input(shape=(5,))
    #split2 = Input(shape=(9,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    x1 = Dense(400, activation='relu')(inputs)
    x2 = Dense(45, activation='relu')(x1)
    x3 = Dense(200, activation='relu')(x2)
    x4 = Dense(45, activation='linear')(x3)
    res_add = Add()([x2, x4])
    x5 = Dense(200, activation='relu')(res_add)
    x6 = Dense(45, activation='linear')(x5)

    model = Model(inputs=inputs, outputs=x6)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model

def build_nn_resnet_fracvol():
    # Batch Size 1000 and 250 epochs, hits a median of 0.8125
    input_dims = 45
    inputs = Input(shape=(input_dims,))

    #split0 = Input(shape=(1,))
    #split1 = Input(shape=(5,))
    #split2 = Input(shape=(9,))

    # Split 0 is 0th order, Split 1 is 2nd order, Split 2 is 4th order
    # split0, split1, split2 = tf.split(inputs, [1, 5, 9], 1)

    # 0th Order Network Flow
    x1 = Dense(400, activation='relu')(inputs)
    x2 = Dense(45, activation='relu')(x1)
    x3 = Dense(200, activation='relu')(x2)
    x4 = Dense(45, activation='linear')(x3)
    res_add = Add()([x2, x4])
    x5 = Dense(200, activation='relu')(res_add)
    x6 = Dense(45, activation='linear')(x5)

    # Extract Fractional Volume Output
    f_out = Dense(3, activation='linear')(x5)

    total_out = concatenate([x6, f_out])
    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=total_out)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss=fracvol_loss, optimizer=opt_func, metrics=[calc_acc, frac_loss, sh_loss])
    print(model.summary())
    return model
