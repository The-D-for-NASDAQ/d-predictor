import numpy as np
from sklearn.utils import shuffle


def make_x_and_y(d, x_block_length, y_block_length):
    d_num_layers = d.shape[0]
    d_num_price_levels = d.shape[1]
    d_total_minutes = d.shape[2]

    highest_bid_position = int(d_num_price_levels / 2)

    X_y_check_entries_count = 0
    X_y_check_entries_pointer = 0
    while X_y_check_entries_pointer + x_block_length + y_block_length < d_total_minutes:
        if d[5, 0, X_y_check_entries_pointer] < d[5, 0, X_y_check_entries_pointer + x_block_length + y_block_length - 1]:
            X_y_check_entries_count += 1
        X_y_check_entries_pointer += 1

    d_pointer = 0
    last_recorded_X_y = 0
    X = np.zeros((X_y_check_entries_count, d_num_layers * d_num_price_levels * x_block_length), np.float32)
    y = np.zeros((X_y_check_entries_count, 3), np.float32)
    while d_pointer + x_block_length + y_block_length < d_total_minutes:
        # slices takes from first but not including last, selecting takes particular!!
        if d[5, 0, d_pointer] > d[5, 0, d_pointer + x_block_length + y_block_length - 1]:
            d_pointer += 1
            continue

        new_X = d[:, :, d_pointer:d_pointer + x_block_length]
        raw_new_y = d[0, highest_bid_position, d_pointer + x_block_length + y_block_length - 1]
        last_X_price = new_X[0, highest_bid_position, -1]

        if raw_new_y - last_X_price > 0:
            new_y = np.array([0, 0, 1], np.float32)
        elif raw_new_y - last_X_price < 0:
            new_y = np.array([1, 0, 0], np.float32)
        else:
            new_y = np.array([0, 1, 0], np.float32)

        X[last_recorded_X_y] = new_X.flatten()
        y[last_recorded_X_y] = new_y

        last_recorded_X_y += 1
        d_pointer += 1

    return X, y


def normalize_data(X, y):
    # normalize data with 0 and non-0 responses
    X_with_minus_one_answer = X[y[:, 0] == 1]
    y_with_minus_one_answer = y[y[:, 0] == 1]
    X_with_zero_answer = X[y[:, 1] == 1]
    y_with_zero_answer = y[y[:, 1] == 1]
    X_with_one_answer = X[y[:, 2] == 1]
    y_with_one_answer = y[y[:, 2] == 1]

    X_with_minus_one_answer, y_with_minus_one_answer = shuffle(X_with_minus_one_answer, y_with_minus_one_answer)
    X_with_zero_answer, y_with_zero_answer = shuffle(X_with_zero_answer, y_with_zero_answer)
    X_with_one_answer, y_with_one_answer = shuffle(X_with_one_answer, y_with_one_answer)

    min_answer_group_length = min(len(y_with_minus_one_answer), len(y_with_zero_answer), len(y_with_one_answer))

    X = np.concatenate((
        X_with_minus_one_answer[:min_answer_group_length],
        X_with_zero_answer[:min_answer_group_length],
        X_with_one_answer[:min_answer_group_length]
    ))
    y = np.concatenate((
        y_with_minus_one_answer[:min_answer_group_length],
        y_with_zero_answer[:min_answer_group_length],
        y_with_one_answer[:min_answer_group_length]
    ))

    return shuffle(X, y)


def make_train_and_test_sets(X, y, desired_test_percentage, desired_test_size):
    # this function will get larger test set from pointer or batch size

    if len(X) == 0:
        exit('X is empty!!!')

    pointer = min(
        len(X) - int(len(X) * desired_test_percentage),
        len(X) - desired_test_size
    )

    X_train = X[:pointer]
    y_train = y[:pointer]
    X_test = X[pointer:]
    y_test = y[pointer:]

    return X_train, y_train, X_test, y_test
