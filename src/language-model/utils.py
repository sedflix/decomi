import numpy as np
from keras import backend as K
from scipy.misc import logsumexp


def sample_logits(preds, temperature=1.0):
    """

    Sample an index from a logit vector.

    :param preds:
    :param temperature:
    :return:
    """
    preds = np.asarray(preds).astype('float64')

    if temperature == 0.0:
        return np.argmax(preds)

    preds = preds / temperature
    preds = preds - logsumexp(preds)

    choice = np.random.choice(len(preds), 1, p=np.exp(preds))


def sparse_loss(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


