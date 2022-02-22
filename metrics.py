import math
import numpy as np
from sklearn.metrics import mean_absolute_error


def get_accuracy(y_test, predictions):
    p_indexes = np.argmax(predictions, axis=1)
    y_indexes = np.argmax(y_test, axis=1)

    true_answers = len(np.where(p_indexes == y_indexes)[0])

    y_sub_1 = np.where(y_test[:, 0] == 1)[0]
    y_0 = np.where(y_test[:, 1] == 1)[0]
    y_1 = np.where(y_test[:, 2] == 1)[0]

    _sub_1_answers = np.where(np.argmax(predictions, axis=1) == 0)[0]
    _0_answers = np.where(np.argmax(predictions, axis=1) == 1)[0]
    _1_answers = np.where(np.argmax(predictions, axis=1) == 2)[0]

    true_sub_1_answers = len(np.intersect1d(_sub_1_answers, y_sub_1))
    true_0_answers = len(np.intersect1d(_0_answers, y_0))
    true_1_answers = len(np.intersect1d(_1_answers, y_1))

    print('Total true answers: ' + str(math.floor(true_answers / len(predictions) * 100)) + '%')
    print('-1 true answers: ' + str(math.floor(true_sub_1_answers / len(_sub_1_answers) * 100)) + '%')
    print('0 true answers: ' + str(math.floor(true_0_answers / len(_0_answers) * 100)) + '%')
    print('1 true answers: ' + str(math.floor(true_1_answers / len(_1_answers) * 100)) + '%')


def get_mean_absolute_error(predictions, y_test):
    return print('Mean absolute error: ' + str(mean_absolute_error(y_test, predictions)))
