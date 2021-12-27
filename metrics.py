import math
import numpy as np
from sklearn.metrics import mean_absolute_error


def get_accuracy(y_test, predictions):
    p_indexes = np.argmax(predictions, axis=1)
    y_indexes = np.argmax(y_test, axis=1)

    true_answers = len(np.where(p_indexes == y_indexes)[0])

    y_sub_1 = len(np.where(y_test[:, 0] == 1)[0])
    y_0 = len(np.where(y_test[:, 1] == 1)[0])
    y_1 = len(np.where(y_test[:, 2] == 1)[0])

    true_sub_1_answers = len(
        np.intersect1d(np.where(np.argmax(y_test, axis=1) == 0)[0], np.where(np.argmax(predictions, axis=1) == 0)[0]))
    true_0_answers = len(
        np.intersect1d(np.where(np.argmax(y_test, axis=1) == 1)[0], np.where(np.argmax(predictions, axis=1) == 1)[0]))
    true_1_answers = len(
        np.intersect1d(np.where(np.argmax(y_test, axis=1) == 2)[0], np.where(np.argmax(predictions, axis=1) == 2)[0]))

    print('Total true answers: ' + str(math.floor(true_answers * 100 / len(predictions))) + '%')
    print('-1 true answers: ' + str(math.floor(true_sub_1_answers * 100 / y_sub_1)) + '%')
    print('0 true answers: ' + str(math.floor(true_0_answers * 100 / y_0)) + '%')
    print('1 true answers: ' + str(math.floor(true_1_answers * 100 / y_1)) + '%')


def get_mean_absolute_error(predictions, y_test):
    return print('Mean absolute error: ' + str(mean_absolute_error(y_test, predictions)))
