import keras.backend as K

def fracvol_loss(y_true, y_pred):

    alpha = 1.0
    beta = 1.0

    fod_loss_term = K.mean(K.square(y_pred[:, :45]-y_true[:, :45]))

    # Please note that Fractional Volume has
    # been hardcoded as the last three values of the vector
    fracvol_loss_term = K.mean(K.square(y_pred[:, 45:48]-y_true[:, 45:48]))

    total_loss = alpha * fod_loss_term + beta * fracvol_loss_term
    return total_loss