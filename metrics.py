import math
import numpy as np
from sklearn.metrics import mean_absolute_error


def get_accuracy(y_test, predictions):
    p_indexes = np.argmax(predictions, axis=1)
    y_indexes = np.argmax(y_test, axis=1)

    true_answers = len(np.where(p_indexes == y_indexes)[0])

    y_up = np.where(y_test[:, 2] == 1)[0]
    y_none = np.where(y_test[:, 1] == 1)[0]
    y_down = np.where(y_test[:, 0] == 1)[0]

    up_answers = np.where(np.argmax(predictions, axis=1) == 2)[0]
    none_answers = np.where(np.argmax(predictions, axis=1) == 1)[0]
    down_answers = np.where(np.argmax(predictions, axis=1) == 0)[0]

    true_up_answers = len(np.intersect1d(up_answers, y_up))
    true_none_answers = len(np.intersect1d(none_answers, y_none))
    true_down_answers = len(np.intersect1d(down_answers, y_down))

    print('Total true answers: ' + str(math.floor(true_answers / len(predictions) * 100)) + '%')
    print('Up true answers: ' + str(math.floor(true_up_answers / len(up_answers) * 100) if len(up_answers) else 0) + '%')
    print('None true answers: ' + str(math.floor(true_none_answers / len(none_answers) * 100) if len(none_answers) else 0) + '%')
    print('Down true answers: ' + str(math.floor(true_down_answers / len(down_answers) * 100) if len(down_answers) else 0) + '%')


def get_mean_absolute_error(predictions, y_test):
    return print('Mean absolute error: ' + str(mean_absolute_error(y_test, predictions)))
