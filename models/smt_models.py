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
from utils.metrics import smt_loss, smt_diff_loss, smt_intra_loss, smt_trans_loss

def build_smt_resnet_fracvol():
    # Batch Size 1000 and 250 epochs, hits a median of 0.8125
    input_dims = 3
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
    #x6 = Dense(45, activation='linear')(x5)

    # Extract Fractional Volume Output
    f_out = Dense(3, activation='linear')(x5)

    #total_out = concatenate([x6, f_out])
    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=f_out)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss=smt_loss, optimizer=opt_func, metrics=[smt_diff_loss, smt_trans_loss, smt_intra_loss])
    print(model.summary())
    return model


def build_smt_patch_resnet_fracvol():

    # Patch Size is hard coded in the network
    #input_dims = (3, 3, 3, 45)
    input_dims = (3, 3, 3, 3)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)
    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:4])(x1)
    sh0_c1 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh0_c2 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh0_c3 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    x2 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c3)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)
    x6 = Dense(3)(x5)

    # Extract Fractional Volume Output
    #f_out = Dense(3, activation='linear')(x5)

    #total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=x6)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss=smt_loss, optimizer=opt_func, metrics=[smt_diff_loss, smt_trans_loss, smt_intra_loss])
    print(model.summary())
    return model



def build_smt_patch_resnet_fracvol_west(weight_para):

    def smt_loss_w_custom(y_true, y_pred):
        ## Breaking down the loss into three terms for weighting
        alpha = weight_para
        diff_term = K.mean(K.square(y_pred[:, 0:1] - y_true[:, 0:1]))
        beta = weight_para
        trans_term = K.mean(K.square(y_pred[:, 1:2] - y_true[:, 1:2]))
        gamma = 1.0
        intra_term = K.mean(K.square(y_pred[:, 2:3] - y_true[:, 2:3]))
        total_loss = alpha * diff_term + beta * trans_term + gamma * intra_term
        return total_loss


    # Patch Size is hard coded in the network
    #input_dims = (3, 3, 3, 45)
    input_dims = (3, 3, 3, 3)
    inputs = Input(shape=input_dims)

    # First Convolution
    x1 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(inputs)
    # Functional Blocks
    sh0 = Lambda(lambda x: x[:, :, :, :, 0:4])(x1)
    sh0_c1 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0)
    sh0_c2 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(sh0_c1)
    sh0_c3 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c2)
    x2 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(sh0_c3)

    # Complete Residual Block
    res_add = Add()([x1, x2])

    x3 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same', activation='relu')(res_add)
    x4 = Conv3D(filters=3, kernel_size=10, strides=(1, 1, 1), padding='same')(x3)

    x5 = Flatten()(x4)
    x6 = Dense(3)(x5)

    # Extract Fractional Volume Output
    #f_out = Dense(3, activation='linear')(x5)

    #total_out = concatenate([x6, f_out])

    # Model define inputs and outputs from network structure
    model = Model(inputs=inputs, outputs=x6)

    opt_func = RMSprop(lr=0.0001)
    model.compile(loss=smt_loss_w_custom, optimizer=opt_func, metrics=[smt_diff_loss, smt_trans_loss, smt_intra_loss])
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