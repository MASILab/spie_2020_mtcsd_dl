from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Dropout, concatenate, Conv3D, Flatten, Lambda
from keras.models import Model
from keras.layers import Activation, Add
from keras.wrappers.scikit_learn import KerasRegressor

from keras.optimizers import SGD, adam, nadam, Adagrad, RMSprop
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, CSVLogger

from utils.metrics import calc_acc, frac_loss, sh_loss

def build_sh_patch_resnet_fracvol():

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)
    x6 = Dense(45)(x5)

    # Extract Fractional Volume Output
    f_out = Dense(3, activation='linear')(x5)

    total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=total_out)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func, metrics=[calc_acc, frac_loss, sh_loss])
    print(model.summary())
    return model

def build_sh_patch_resnet():

    # Patch Size is hard code in the network
    input_dims = (3, 3, 3, 45)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)

    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:1])(x1)
    sh2 = Lambda(lambda x: x[:, :, :, :, 1:6])(x1)
    sh4 = Lambda(lambda x: x[:, :, :, :, 6:15])(x1)
    sh6 = Lambda(lambda x: x[:, :, :, :, 16:28])(x1)
    sh8 = Lambda(lambda x: x[:, :, :, :, 28:45])(x1)

    sh0_c1 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh2_c1 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2)
    sh4_c1 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4)
    sh6_c1 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6)
    sh8_c1 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8)

    sh0_c2 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh2_c2 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh2_c1)
    sh4_c2 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh4_c1)
    sh6_c2 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh6_c1)
    sh8_c2 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh8_c1)

    sh0_c3 = Conv3D(filters=1, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    sh2_c3 = Conv3D(filters=5, kernel_size=10, strides=(1, 1, 1), padding='same')(sh2_c2)
    sh4_c3 = Conv3D(filters=9, kernel_size=10, strides=(1, 1, 1), padding='same')(sh4_c2)
    sh6_c3 = Conv3D(filters=13, kernel_size=10, strides=(1, 1, 1), padding='same')(sh6_c2)
    sh8_c3 = Conv3D(filters=17, kernel_size=10, strides=(1, 1, 1), padding='same')(sh8_c2)

    combined = concatenate([sh0_c3, sh2_c3, sh4_c3, sh6_c3, sh8_c3])
    x2 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(combined)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=45, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)
    x6 = Dense(45)(x5)
    #x2 = Conv3D(filters=1, kernel_size=1, strides=(1, 1, 1), padding='same')(x1)
    model = Model(inputs=inputs, outputs=x6)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=opt_func)
    print(model.summary())
    return model


def build_nn_resnet():

    # Remember to tune the bias and kernel initializers for the convolutions once the network is complete
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