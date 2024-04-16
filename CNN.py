import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

def train_CNN(X_train, y_train, X_test, y_test):
    print("Building CNN...")

    cnn = Sequential()
    cnn.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1,)))
    cnn.add(MaxPooling1D((2), padding="same"))
    cnn.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling1D((2)))
    cnn.add(Flatten())
    cnn.add(Dense(units=32, activation='relu'))
    cnn.add(Dense(y_train.shape[1], activation='softmax'))
    cnn.summary()

    cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training CNN...")
    hist = cnn.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

    train_acc = hist.history['accuracy'][-1]
    print("CNN train acc: ", train_acc)

    results = model.evaluate(x_test, y_test_prepared, batch_size=32)
    print("CNN test loss, test acc:", results)

    return results