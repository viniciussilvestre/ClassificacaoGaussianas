import matplotlib
import tensorflow as tf
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import *

matplotlib.rcParams['figure.figsize'] = [9, 6]

seed = 22
tf.random.set_seed(seed)
random.seed(a=seed)
batch_size = 32


def read_data():
    with open("DATA.txt") as f:
        lines = f.readlines()

    input_data = np.empty(shape=(lines.__len__(), 2))
    for i in range(lines.__len__()):
        raw_data = lines[i].split(",")
        data0 = float(raw_data[0])
        data1 = float(raw_data[1])
        input_data[i][0] = data0
        input_data[i][1] = data1

    with open("TARGETS.txt") as f:
        lines = f.readlines()

    target_data = []
    for j in range(lines.__len__()):
        raw_line = lines[j].split(",")
        raw_data = []
        for k in range(raw_line.__len__()):
            raw_data.append(int(raw_line[k]))
        target_data.append(raw_data)

    return input_data, np.array(target_data)


def plot_data(data, title):
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    for aux in range(data.__len__()):
        points = data[aux]
        plt.plot(points[0], points[1], marker="o", markersize=5, markeredgecolor=points[2], markerfacecolor=points[2])
    # plt.scatter(x_data, y_data)
    plt.show()


def read_input():
    stop = "val_loss"
    print("Please insert the learning rate, percentage of data reserved for testing, percentage of data for validation, max epochs, neurons in hidden layer, early stop parameter\n")
    print("Learning rate: ")
    try:
        learning = float(input())
    except ValueError:
        raise TypeError("Expected Float, got wrong type!\n")
    print("Percentage of data reserved for training: (0.x)")
    try:
        reserved = float(input())
    except ValueError:
        raise TypeError("Expected Float, got wrong type!\n")
    print("Percentage of data reserved for validation: (0.x)")
    try:
        reserved_validation = float(input())
    except ValueError:
        raise TypeError("Expected Float, got wrong type!\n")
    print("Max epochs: ")
    try:
        epochs = int(input())
    except ValueError:
        raise TypeError("Expected Int, got wrong type!\n")
    print("Neurons in hidden layer: ")
    try:
        neurons = int(input())
    except ValueError:
        raise TypeError("Expected Int, got wrong type!\n")
    # print("Stop condition: 1 validation loss (val_loss), 2 accuracy (accuracy), 3 training loss (loss)")
    print("Stop condition: validation loss (val_loss), accuracy (accuracy), training loss (loss)")
    stop = input()

    # try:
    #     stop_int = int(input())
    # except ValueError:
    #     raise TypeError("Expected Int, got wrong type!\n")
    #
    # # if stop_int == 1:
    # #     stop = "val_loss"
    # # elif stop_int == 2:
    # #     stop = "accuracy"
    # # elif stop_int == 3:
    # #     stop = "loss"

    return learning, reserved, epochs, reserved_validation, neurons, stop


def select_training_data(data_to_sort_through, target_data_to_sort_through, samples):
    result = np.empty(shape=(samples, 2))
    target_result = np.empty(shape=(samples, 3))
    data_not_chosen = np.empty(shape=((data_to_sort_through.shape[0] - samples), 2))
    target_data_not_chosen = np.empty(shape=((data_to_sort_through.shape[0] - samples), 3))
    index_of_data_chosen = []
    for i in range(samples):
        new = True
        while new:
            line = random.randrange(0, data_to_sort_through.__len__() - 1, 1)
            possible_new_element0 = data_to_sort_through[line][0]
            possible_new_element1 = data_to_sort_through[line][1]
            temp = [possible_new_element0, possible_new_element1]
            if temp not in result:
                result[i][0] = possible_new_element0
                result[i][1] = possible_new_element1
                element = target_data_to_sort_through[line]
                target_result[i][0] = element[0]
                target_result[i][1] = element[1]
                target_result[i][2] = element[2]
                new = False
                index_of_data_chosen.append(line)
    aux = 0
    for j in range(data_to_sort_through.shape[0]):
        if j not in index_of_data_chosen:
            data_not_chosen[aux][0] = data_to_sort_through[j][0]
            data_not_chosen[aux][1] = data_to_sort_through[j][1]
            element = target_data_to_sort_through[j]
            target_data_not_chosen[aux][0] = element[0]
            target_data_not_chosen[aux][1] = element[1]
            target_data_not_chosen[aux][2] = element[2]
            aux += 1

    return np.array(result), np.array(target_result), np.array(data_not_chosen), np.array(target_data_not_chosen)


def prepare_data_for_plot(x_data, y_data):
    data_to_plot = []
    for u in range(x_data.__len__()):
        point = [x_data[u][0], x_data[u][1]]
        if y_data[u] == 0:
            point.append('green')
        elif y_data[u] == 1:
            point.append('red')
        elif y_data[u] == 2:
            point.append('blue')
        data_to_plot.append(point)
    return data_to_plot


def create_dataset(train_amount):
    data, target = read_data()

    train_test_split = int(train_amount * data.__len__())
    raw_train, raw_train_target, raw_test, raw_test_target = select_training_data(data, target, train_test_split)

    raw_train_target_list = []
    raw_test_target_list = []

    for aux in range(raw_train_target.__len__()):
        if np.array_equal(raw_train_target[aux], [1, 0, 0]):
            raw_train_target_list.append(0)
        elif np.array_equal(raw_train_target[aux], [0, 1, 0]):
            raw_train_target_list.append(1)
        elif np.array_equal(raw_train_target[aux], [0, 0, 1]):
            raw_train_target_list.append(2)
    raw_train_target_np = np.array(raw_train_target_list)

    for aux in range(raw_test_target.__len__()):
        if np.array_equal(raw_test_target[aux], [1, 0, 0]):
            raw_test_target_list.append(0)
        elif np.array_equal(raw_test_target[aux], [0, 1, 0]):
            raw_test_target_list.append(1)
        elif np.array_equal(raw_test_target[aux], [0, 0, 1]):
            raw_test_target_list.append(2)
    raw_test_target_np = np.array(raw_test_target_list)

    return raw_train, raw_train_target_np, raw_test, raw_test_target_np


def plot_metric(history_data, metric):
    train_metrics = history_data.history[metric]
    for t in range(train_metrics.__len__()):
        train_metrics[t] = 1 - train_metrics[t]
    val_metrics = history_data.history['val_' + metric]
    for t in range(val_metrics.__len__()):
        val_metrics[t] = 1 - val_metrics[t]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation ' + 'errors')
    plt.xlabel("Epochs")
    plt.ylabel('Percentage of errors')
    plt.legend(["train_" + 'errors', 'val_' + 'errors'])
    plt.show()


def plot_matrix(matrix):
    plot_confusion_matrix(matrix)
    plt.show()


def main():
    # Asking inputs from user
    learning_rate, percentage_reserved, max_epochs, reserved_for_validation, hidden_neurons, stop_condition = read_input()

    # Separating data for training, validation and testing
    train_x, train_y, test_x, test_y = create_dataset(percentage_reserved)
    plot_data(prepare_data_for_plot(train_x, train_y), "Training data")
    plot_data(prepare_data_for_plot(test_x, test_y), "Testing data")

    # Creating the MLP
    callback = tf.keras.callbacks.EarlyStopping(monitor=stop_condition, patience=8)
    inputs = tf.keras.Input(shape=(2,), name='coordinates')
    x = tf.keras.layers.Dense(hidden_neurons, activation='relu', name='hidden_1')(inputs)
    outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    # Collecting the results
    history = model.fit(train_x, train_y, batch_size=32, epochs=max_epochs, callbacks=callback, validation_split=reserved_for_validation, use_multiprocessing=True)
    test_results = model.evaluate(test_x, test_y, use_multiprocessing=True)
    predictions = model.predict(test_x)

    # Showing the results
    print(test_results)
    plot_metric(history, 'accuracy')
    result = confusion_matrix(test_y, predictions.argmax(axis=1))
    plot_matrix(result)


if __name__ == '__main__':
    main()
