import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization

def train_DNN(X_train, y_train, X_test, y_test):
    print("Building DNN...")
    dnn = Sequential()
    dnn.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.2))
    dnn.add(Dense(64, activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.2))
    dnn.add(Dense(32, activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(0.2))
    dnn.add(Dense(y_train.shape[1], activation='softmax'))
    dnn.summary()

    dnn.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training DNN...")
    hist = dnn.fit(X_train, y_train, epochs=60, validation_split=0.2)

    train_acc = hist.history['acc'][-1]
    print("DNN Train acc: ", train_acc)

    results = dnn.evaluate(X_test, y_test, batch_size=128)
    print("DNN test loss, test acc:", results)

    return results