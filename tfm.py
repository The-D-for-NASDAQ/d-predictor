import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

best_checkpoint_path = os.path.join('checkpoints', 'best_weights.hdf5')


def lrelu(x):
    return tf.nn.leaky_relu(x, alpha=0.01)


def create_model():
    model = Sequential()

    model.add(Dense(1630, activation=lrelu))
    model.add(Dense(1630, activation=lrelu))
    model.add(Dense(400, activation=lrelu))
    model.add(Dense(50, activation=lrelu))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def fit_model(model, batch_size, epochs, X_train, y_train, X_test, y_test):
    train_data_slice = (len(X_train) // batch_size) * batch_size
    test_data_slice = (len(X_test) // batch_size) * batch_size

    try:
        model.fit(x=X_train[:train_data_slice], y=y_train[:train_data_slice],
                  batch_size=batch_size, epochs=epochs,
                  validation_data=(X_test[:test_data_slice], y_test[:test_data_slice]),
                  callbacks=[
                      ModelCheckpoint(best_checkpoint_path, monitor='val_accuracy',
                                      save_best_only=True, verbose=1, save_weights_only=True),
                      TensorBoard(log_dir='logs', histogram_freq=10)
                  ],
                  )
    except KeyboardInterrupt:
        return model

    return model


def load_best_weights(model):
    model.load_weights(best_checkpoint_path)

    return model
