import tensorflow as tf
import keras.backend as K
import numpy as np

######################################
### SMT Losses
######################################
def smt_loss(y_true, y_pred):
    ## Breaking down the loss into three terms for weighting
    # Weight of 1020 was derived from step grid search
    alpha = 1020.0
    diff_term = K.mean(K.square(y_pred[:, 0:1] - y_true[:, 0:1]))
    beta = 1020.0
    trans_term = K.mean(K.square(y_pred[:, 1:2] - y_true[:, 1:2]))
    gamma = 1.0
    intra_term = K.mean(K.square(y_pred[:, 2:3] - y_true[:, 2:3]))
    total_loss = alpha * diff_term + beta * trans_term + gamma * intra_term
    return total_loss

def smt_diff_loss(y_true, y_pred):
    ## Breaking down the loss into three terms for weighting
    alpha = 1020.0
    diff_term = K.mean(K.square(y_pred[:, 0:1] - y_true[:, 0:1]))
    total_diff = alpha * diff_term
    return total_diff

def smt_trans_loss(y_true, y_pred):
    ## Breaking down the loss into three terms for weighting
    beta = 1020.0
    trans_term = K.mean(K.square(y_pred[:, 1:2] - y_true[:, 1:2]))
    total_trans = beta * trans_term
    return total_trans

def smt_intra_loss(y_true, y_pred):
    ## Breaking down the loss into three terms for weighting
    gamma = 1.0
    intra_term = K.mean(K.square(y_pred[:, 2:3] - y_true[:, 2:3]))
    total_intra = gamma * intra_term
    return total_intra


######################################
### SMT Losses End ###################
######################################

def calc_acc(y_true, y_pred):
    # Normalize each vector
    y_true = y_true[0:45]
    y_pred = y_pred[0:45]
    comp_true = tf.conj(y_true)
    norm_true = y_true / tf.sqrt(tf.reduce_sum(tf.multiply(y_true, comp_true)))

    comp_pred = tf.conj(y_pred)
    norm_pred = y_pred / tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, comp_pred)))

    comp_p2 = tf.conj(norm_pred)
    acc = tf.real(tf.reduce_sum(tf.multiply(norm_true, comp_p2)))
    return acc

def calc_acc_numpy(y_true, y_pred):
    y_true = y_true[0:45]
    y_pred = y_pred[0:45]
    comp_true = np.conj(y_true)
    norm_true = y_true / np.sqrt(np.sum(np.multiply(y_true, comp_true)))

    comp_pred = np.conj(y_pred)
    norm_pred = y_pred / np.sqrt(np.sum(np.multiply(y_pred, comp_pred)))

    comp_p2 = np.conj(norm_pred)
    acc = np.real(np.sum(np.multiply(norm_true, comp_p2)))
    return acc

def frac_loss(y_true, y_pred):

    fracvol_loss_term = K.mean(K.square(y_pred[:, 45:48] - y_true[:, 45:48]))
    return fracvol_loss_term

def sh_loss(y_true, y_pred):
    fod_loss_term = K.mean(K.square(y_pred[:, :45] - y_true[:, :45]))
    return fod_loss_term