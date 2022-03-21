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
        check_time_entries = d[5, 0, X_y_check_entries_pointer:X_y_check_entries_pointer + x_block_length + y_block_length]
        check_price_entries = d[0, highest_bid_position-1:highest_bid_position+1, X_y_check_entries_pointer:X_y_check_entries_pointer + x_block_length + y_block_length]
        check_price_difference = check_price_entries[0] - check_price_entries[1]

        if check_time_entries[0] < check_time_entries[-1] and np.any(check_price_difference >= 0) and np.any(check_price_difference <= 0.5):
            X_y_check_entries_count += 1

        X_y_check_entries_pointer += 1

    d_pointer = 0
    last_recorded_X_y = 0
    X = np.zeros((X_y_check_entries_count, (d_num_layers - 2) * d_num_price_levels * x_block_length + 20 + 10), np.float32)
    y = np.zeros((X_y_check_entries_count, 3), np.float32)
    while d_pointer + x_block_length + y_block_length < d_total_minutes:
        time_entries = d[5, 0, d_pointer:d_pointer + x_block_length + y_block_length]
        price_entries = d[0, highest_bid_position-1:highest_bid_position+1, d_pointer:d_pointer + x_block_length + y_block_length]
        price_difference = price_entries[0] - price_entries[1]

        if time_entries[0] > time_entries[-1] or np.any(price_difference < 0) or np.any(price_difference > 0.5):
            d_pointer += 1
            continue

        new_X = d[1:5, :, d_pointer:d_pointer + x_block_length]
        raw_new_y = price_entries[1, -1]
        last_X_price = price_entries[1, -1-y_block_length]

        if raw_new_y > last_X_price:
            new_y = np.array([0, 0, 1], np.float32)
        elif raw_new_y < last_X_price:
            new_y = np.array([1, 0, 0], np.float32)
        else:
            new_y = np.array([0, 1, 0], np.float32)

        X[last_recorded_X_y] = np.concatenate((
            new_X.flatten(),

            price_entries[0, :-y_block_length] - price_entries[0, 0],
            price_entries[1, :-y_block_length] - price_entries[1, 0],
            time_entries[:-y_block_length]
        ))
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

    X = np.concatenate((
        np.repeat(X_with_minus_one_answer, 12, axis=0),
        X_with_zero_answer,
        np.repeat(X_with_one_answer, 12, axis=0)
    ))
    y = np.concatenate((
        np.repeat(y_with_minus_one_answer, 12, axis=0),
        y_with_zero_answer,
        np.repeat(y_with_one_answer, 12, axis=0)
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
