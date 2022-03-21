import glob
import numpy as np
import re
from datetime import datetime, timedelta
from keras.models import load_model
from tfm import lrelu


class bcolors:
    BOLD = '\33[1m'
    UPGREEN = '\33[102m' + BOLD
    DOWNRED = '\33[101m' + BOLD
    ENDC = '\033[0m'


last_price = 0
last_p_index = None

new_price = 0


predict_x_by_minutes = 10

d_num_layers = 6  # Price, Ordered volume, Filled volume, Canceled volume, Pending volume, Time index
d_num_price_levels = 10 * 2 * 2  # price level ($10) per 50 cents per level (*2) per side (*2)

highest_bid_position = int(d_num_price_levels / 2)

model = load_model('models/best_model.h5', custom_objects={'lrelu': lrelu})


def load_data(to_date):
    npy_data_path = '../d-compressor/data/processed_minutes/'+to_date.strftime('%Y%m%d')+'/AAPL/*.npy'
    files = sorted(glob.glob(npy_data_path))

    x_from_time = (to_date - timedelta(minutes=predict_x_by_minutes)).strftime('%H%M')

    files_to_load = []
    for file in files:
        file_time = re.search(r'\/([0-9]{4}).npy', file).group(1)
        if x_from_time < file_time <= to_date.strftime('%H%M'):
            files_to_load.append(file)

    if len(files_to_load) != predict_x_by_minutes:
        return

    data = np.zeros((d_num_layers, d_num_price_levels, predict_x_by_minutes), np.float32)

    load_pointer = 0
    for file in files_to_load:
        data[:, :, load_pointer:load_pointer+1] = np.load(file)
        load_pointer += 1

    return data


def validate_and_convert_data(data):
    global new_price

    price_entries = data[0, highest_bid_position-1:highest_bid_position+1]
    price_difference = price_entries[0] - price_entries[1]

    new_price = price_entries[1, -1]

    if np.any(price_difference < 0) or np.any(price_difference > 0.5):
        return

    new_data = np.zeros((1, (d_num_layers - 2) * d_num_price_levels * predict_x_by_minutes + 20 + 10), np.float32)

    new_data[0] = np.concatenate((
        data[1:5].flatten(),

        price_entries[0] - price_entries[0, 0],
        price_entries[1] - price_entries[1, 0],
        data[5, 0]
    ))

    return new_data


def main(date_to_process):
    global last_price, last_p_index

    begin_time = datetime.now()

    data = load_data(date_to_process)

    if data is None:
        return

    converted_data = validate_and_convert_data(data)

    if converted_data is None:
        return

    prediction = model.predict(converted_data)
    p_index = np.argmax(prediction)

    if last_price and last_p_index is not None:
        if last_p_index == 0:
            if new_price < last_price:
                print(bcolors.UPGREEN + 'Down ▼ prediction was correct' + bcolors.ENDC)
            else:
                print(bcolors.DOWNRED + 'Down ▼ prediction was incorrect' + bcolors.ENDC)

        if last_p_index == 2:
            if new_price > last_price:
                print(bcolors.UPGREEN + 'Up ▲ prediction was correct' + bcolors.ENDC)
            else:
                print(bcolors.DOWNRED + 'Up ▲ prediction was incorrect' + bcolors.ENDC)

    last_price = new_price
    last_p_index = p_index

    predicted_direction = 'None'
    if p_index == 0:
        predicted_direction = bcolors.DOWNRED + 'Down ▼' + bcolors.ENDC
    elif p_index == 2:
        predicted_direction = bcolors.UPGREEN + 'Up ▲' + bcolors.ENDC

    print('At: ' + str(datetime.now(tz=date_to_process.tzinfo)) +
          ' | Predicted date: ' + str(date_to_process) +
          ' | Predicting time: ' + str(datetime.now() - begin_time) +
          ' | Predicted direction: ' + str(predicted_direction)
          )
