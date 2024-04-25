import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from preprocessing import csv_to_xy


def create_DNN(X_train, y_train):
    print("Compiling DNN...")
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

    return dnn


def train_DNN(train_file, test_file, instances=5, epochs=60):

    X_train, y_train = csv_to_xy(train_file)
    X_test, y_test = csv_to_xy(test_file)

    best_test_acc = 0
    best_hist = None
    best_dnn = None

    for i in range(instances):
        dnn = create_DNN(X_train, y_train)

        print(f"Training DNN {i}...\n")
        hist = dnn.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
        test_acc = dnn.evaluate(X_test, y_test, batch_size=128)[1]
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_hist = hist
            best_dnn = dnn

    # save model
    best_dnn.save(f'saved_models/dnn_{X_train.shape[0]}.keras')
    
    log_file = f'logs/DNN_{X_train.shape[0]}_result.txt'
    # write training results to log file
    with open(log_file, 'w') as file:
        file.write(f"{train_file} {test_file}\n\n")
        file.write("Train accuracy, Val accuracy\n")
        for i in range(len(best_hist.history['accuracy'])):
            file.write(f"Epoch {i+1}: {best_hist.history['accuracy'][i]} {best_hist.history['val_accuracy'][i]}\n")
        file.write(f"\nDNN Final train acc: {best_hist.history['accuracy'][-1]}\n")
        file.write(f"DNN Test acc: {best_test_acc}")

    plot_file = f'plots/DNN_{X_train.shape[0]}_training.png'
    plt.clf()
    plt.plot(best_hist.history['accuracy'], label='trn_accuracy')
    plt.plot(best_hist.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('DNN Training and Validation Accuracy')
    plt.savefig(plot_file)

    print(f"DNN log saved to {log_file} and plot saved to {plot_file}\n")


def create_CNN(X_train, y_train):
    print("Compiling CNN...")
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

    return cnn


def train_CNN(train_file, test_file, instances=5, epochs=60):

    X_train, y_train = csv_to_xy(train_file)
    X_test, y_test = csv_to_xy(test_file)

    best_test_acc = 0
    best_hist = None
    best_cnn = None

    for i in range(instances):
        cnn = create_CNN(X_train, y_train)

        print(f"Training CNN {i}...\n")
        hist = cnn.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
        test_acc = cnn.evaluate(X_test, y_test, batch_size=128)[1]
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_hist = hist
            best_cnn = cnn

    # save model
    best_cnn.save(f'saved_models/cnn_{X_train.shape[0]}.keras')

    log_file = f'logs/CNN_{X_train.shape[0]}_result.txt'
    # write training results to log file
    with open(log_file, 'w') as file:
        file.write(f"{train_file} {test_file}\n\n")
        file.write("Train accuracy, Val accuracy\n")
        for i in range(len(best_hist.history['accuracy'])):
            file.write(f"Epoch {i+1}: {best_hist.history['accuracy'][i]} {best_hist.history['val_accuracy'][i]}\n")
        file.write(f"\nCNN Final train acc: {best_hist.history['accuracy'][-1]}\n")
        file.write(f"CNN Test acc: {best_test_acc}")

    plot_file = f'plots/CNN_{X_train.shape[0]}_training.png'
    plt.clf()
    plt.plot(best_hist.history['accuracy'], label='trn_accuracy')
    plt.plot(best_hist.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('CNN Training and Validation Accuracy')
    plt.savefig(plot_file)

    print(f"CNN log saved to {log_file} and plot saved to {plot_file}\n")


def create_CNN_LSTM(X_train, y_train):
    print("Compiling CNN-LSTM...")
    cnnlstm = Sequential()
    cnnlstm.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],1,)))
    cnnlstm.add(MaxPooling1D((2), padding="same"))
    cnnlstm.add(Conv1D(64, kernel_size=3, activation='relu'))
    cnnlstm.add(MaxPooling1D((2)))
    cnnlstm.add(LSTM(80))
    cnnlstm.add(Dense(units=64, activation='relu'))
    cnnlstm.add(Dense(y_train.shape[1], activation='softmax'))
    cnnlstm.summary()

    cnnlstm.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return cnnlstm


def train_CNN_LSTM(train_file, test_file, instances=5, epochs=60):

    X_train, y_train = csv_to_xy(train_file)
    X_test, y_test = csv_to_xy(test_file)

    best_test_acc = 0
    best_hist = None
    best_cnnlstm = None

    for i in range(instances):
        cnnlstm = create_CNN_LSTM(X_train, y_train)

        print(f"Training CNN-LSTM {i}...\n")
        hist = cnnlstm.fit(X_train, y_train, epochs=epochs, validation_split=0.2)
        test_acc = cnnlstm.evaluate(X_test, y_test, batch_size=128)[1]
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_hist = hist
            best_cnnlstm = cnnlstm

    # save model
    best_cnnlstm.save(f'saved_models/cnnlstm_{X_train.shape[0]}.keras')

    log_file = f'logs/CNN-LSTM_{X_train.shape[0]}_result.txt'
    # write training results to log file
    with open(log_file, 'w') as file:
        file.write(f"{train_file} {test_file}\n\n")
        file.write("Train accuracy, Val accuracy\n")
        for i in range(len(best_hist.history['accuracy'])):
            file.write(f"Epoch {i+1}: {best_hist.history['accuracy'][i]} {best_hist.history['val_accuracy'][i]}\n")
        file.write(f"\nCNN-LSTM  Final train acc: {best_hist.history['accuracy'][-1]}\n")
        file.write(f"CNN-LSTM Test acc: {best_test_acc}")

    plot_file = f'plots/CNN-LSTM_{X_train.shape[0]}_training.png'
    plt.clf()
    plt.plot(best_hist.history['accuracy'], label='trn_accuracy')
    plt.plot(best_hist.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('CNN-LSTM Training and Validation Accuracy')
    plt.savefig(plot_file)

    print(f"CNN-LSTM log saved to {log_file} and plot saved to {plot_file}\n")


def evaluate_saved_model(model_path, test_file):

    X_test, y_test = csv_to_xy(test_file)

    model = load_model(model_path)

    return model.evaluate(X_test, y_test, batch_size=128)[1]
