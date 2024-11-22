import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras as ks
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import sys
from io import StringIO
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

def setupdf(file_path):
    """
    Function that reads the text file in the filepath, sets the first columns name to 'label' and sets all other columns
    name to their equivalent timestamp.

    :param file_path: filepath for the setup file
    :return: A pandas dataframe with renamed columns
    """

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    for i in range(1, len(df.columns)):
        df.rename(columns={i: i}, inplace=True)
    # print(df)
    return df


def setupdfs(df):
    """
    Takes a pandas dataframe and splits it according to label. Label 1-5 each become their own dataframe. Dataframes
    will be named after their heart problem.

    :param df: Pandas Dataframe
    :return: 5 different dataframes which are the five types of heartbeats in the dataset.
    """
    healthy_df = df[df['label'] == 1.0]
    arrhythmia_df = df[df['label'] == 2.0]
    prematureV_df = df[df['label'] == 3.0]
    prematureA_df = df[df['label'] == 4.0]
    other_df = df[df['label'] == 5.0]
    return [healthy_df, arrhythmia_df, prematureV_df, prematureA_df, other_df]


def plot_df(dataframe, num, title):
    """
    Utility function that plots heartbeats from a dataframe.

    :param dataframe: Dataframe to plot
    :param num: Number of heartbeats to plot starting from the oth heartbeat.
    :param title: What to print on top of the table.
    """
    plt.figure(figsize=(12, 6))
    num_heartbeats_to_plot = num
    for index, heartbeat in dataframe.iloc[:num_heartbeats_to_plot, 1:].iterrows():  # Skip the label column
        plt.plot(range(len(heartbeat.values)), heartbeat.values, label=f'Heartbeat {index + 1}')

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Pulse Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()


def plotNumGraph(data_array, num, title):
    """
    Plots the heartbeat in a certain position in the data array.
    :param data_array: The data array to plot from.
    :param num: The heartbeat to plot (remember that python's initial index is zero!)
    :param title: What to pot on top of the plot
    :return: ---
    """
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data_array[0])), data_array[num], label=f'Heartbeat {num + 1}')
    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Pulse Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()


def plot_eachtype(dataframes):
    """
    Plots each type of heart beat.
    :param dataframes: Array of dataframes to plot from.
    :return: ---
    """
    labels = ["Healthy Heartbeat", "Arrhythmia Heartbeat", "Premature Ventricular", "Premature Artrial", "Others"]

    plt.figure(figsize=(12, 6))

    for i, (df, label) in enumerate(zip(dataframes, labels)):
        # Plot only one heartbeat from each DataFrame (first row)
        heartbeat = df.iloc[0, 1:]  # Skips the label column
        plt.plot(range(len(heartbeat)), heartbeat.values, label=label)

    plt.title('All 5 Types of Heartbeats')
    plt.xlabel('Time Index')
    plt.ylabel('Pulse Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()  # Display all plots in one figure


def createModel(epochs, batch_size, time_steps, features, model, train_data):
    """
    Uses Keras to build the autoencoder.
    :param epochs: Number of times the model will iterate over the entire training dataset
    :param batch_size: Defines the number of samples used in one forward/backward pass during training
    :param time_steps: Determines the number of discrete intervals the model uses to analyze the heartbeat
    :param features: The amount of features used per time step. time_step*feature = length of heartbeat
    :return: History which stores the training loss for each epoch as well as the validation loss.
    """
    # Encoder
    model.add(LSTM(64, activation='tanh', input_shape=(time_steps, features), return_sequences=True))
    #    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(32, activation='tanh', return_sequences=True))

    # Bottleneck
    model.add(LSTM(10, activation='tanh', return_sequences=True))  # Smallest representation layer (bottleneck)

    # Decoder
    model.add(LSTM(32, activation='tanh', return_sequences=True))
    model.add(LSTM(64, activation='tanh', return_sequences=True))
    model.add(LSTM(140, activation='tanh'))  # Output layer supports negative values

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(train_data, train_data, batch_size, epochs, validation_split=0.1)
    return history


def plotReconstructed(num_heartbeat, epochs, batch_size, history, model, modelnumber, scaler):
    """
    Uses the created model to reconstruct a heartbeat.

    :param num_heartbeat: The heartbeat to reconstruct. This number of heartbeat will be pulled from the healthy heartbeat dataframe
    :param epochs: Number of epochs used for training
    :param batch_size: Batch size used for training
    :param history: history of the training process
    :return: ---
    """
    # Select a single heartbeat from the input data
    input_heartbeat = healthy_norm[num_heartbeat].reshape(1, 140, 1)  # input_heartbeat is a NumPy array with shape (1, 140, 1)

    # Get the reconstructed heartbeat from the model and reverse scale it
    reconstructed_heartbeat = model.predict(input_heartbeat)  # reconstructed_heartbeat is a NumPy array with shape (1, 140, 1)
    reconstructed_heartbeat_inverse_scaled = scaler.inverse_transform(reconstructed_heartbeat)  # reconstructed_heartbeat_inverse_scaled is a NumPy array with shape (1, 140)

    input_heartbeat_inverse_scaled = scaler.inverse_transform([healthy_norm[num_heartbeat]])  # input_heartbeat_inverse_scaled is a NumPy array with shape (1, 140)

    mse = mean_squared_error(input_heartbeat_inverse_scaled, reconstructed_heartbeat_inverse_scaled) # calculate the error between the unscaled versions

    # Plotting both the input and reconstructed heartbeat for the unscaled version
    plt.figure(figsize=(10, 5))
    plt.plot(input_heartbeat.reshape(-1), label=f"Input Heartbeat {num_heartbeat}", linestyle='--', color='blue')  # input_heartbeat reshaped to (140,)
    plt.plot(reconstructed_heartbeat[0], label=f"Reconstructed Heartbeat Model {modelnumber}", linestyle='-', color='red')  # reconstructed_heartbeat[0] reshaped to (140,)
    plt.title(
        f"Comparison of Input and Reconstructed Heartbeat. Final loss is: {round(history.history['loss'][-1], 5)} \n"
        f"Epochs = {epochs}, Batch Size = {batch_size}")
    plt.xlabel("Timestep")
    plt.ylabel("Pulse Value")
    plt.legend()
    plt.show()


    # Plotting both the input and reconstructed heartbeat
    plt.figure(figsize=(10, 5))
    plt.plot(input_heartbeat_inverse_scaled.reshape(-1), label=f"Input Heartbeat {num_heartbeat}", linestyle='--', color='blue')
    plt.plot(reconstructed_heartbeat_inverse_scaled.reshape(-1), label=f"Reconstructed Heartbeat Model {modelnumber}", linestyle='-',
             color='red')
    plt.title(
        f"Comparison of Input and Reconstructed Heartbeat after reverse scaling. Final loss is: {round(mse, 5)} \n"
        f"Epochs = {epochs}, Batch Size = {batch_size}")
    plt.xlabel("Timestep")
    plt.ylabel("Pulse Value")
    plt.legend()
    plt.show()


def get_current_modelnumber():
    """
    Gets the current modelnumber from a text file
    :return: The current model number as an int
    """
    value = 0
    with open('modelnumber.txt', 'r') as f:
        value = int(f.read())
    return value


def increase_modelnumber():
    """
    Opens the modelnumber file and increases its value by one.
    :return: ---
    """
    value = 0
    with open('modelnumber.txt', 'r') as f:
        value = int(f.read())
    value += 1
    with open('modelnumber.txt', 'w') as f:
        f.write(str(value))


def document_model(modelnumber, model, epochs, batch_size, history):
    """
    Documents the current model. It creates a text file according to the current modelnumber and stores information
    about the current model in it.

    :param modelnumber: current modelnumber
    :param model: current model to store information about
    :param epochs: numbers of epochs used for training
    :param batch_size: batch size used for training
    :param history: stores losses of the model
    :return: ---
    """
    with open(f"modelnumber{modelnumber}_documentation.txt", "w") as f:
        f.write(f"Training: Epochs = {epochs}, Batch size = {batch_size}, \n"
                f"Result: Loss = {round(history.history['loss'][-1], 5)} \n")

        # Capture model.summary() output
        old_stdout = sys.stdout  # Save the current stdout
        sys.stdout = StringIO()  # Redirect stdout to a string buffer
        model.summary()  # Print the summary
        summary_str = sys.stdout.getvalue()  # Get the summary as a string
        sys.stdout = old_stdout  # Reset stdout to its original value

        # Write the model summary to the file
        f.write("Model Summary:\n")
        f.write(summary_str)


def calculate_mean(model, test_data):
    """
    Calculates the mean mse value for the model predicting the test_data
    :param model:
    :param test_data:
    :return:  mean losses and max loss
    """
    losses = []
    for i in test_data:
        input_heartbeat = i.reshape(1, 140, 1)
        reconstructed_heartbeat = model.predict(input_heartbeat)
        losses.append(mean_squared_error(reconstructed_heartbeat, input_heartbeat))
    losses_np = np.array(losses)
    return np.mean(losses_np), np.max(losses_np)


train_df = setupdf("ECG5000_TRAIN.txt")
test_df = setupdf("ECG5000_TEST.txt")

# combining train and test dataframe
combineddf = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# splitting according to the different labels
[healthy_df, arrhythmia_df, prematureV_df, prematureA_df, other_df] = setupdfs(combineddf)
# print(healthy_df.shape, arrhythmia_df.shape, prematureV_df.shape, prematureA_df.shape, other_df.shape)
# shapes: 2919,141 1767,141 96, 141 194,141 24,141

# reshaping for further use, after it will be a 2D matrix.
# splitting healthy_df into 80%-20% train and test split
healthy_train = healthy_df.iloc[583:, 1:].values
healthy_test = healthy_df.iloc[:582, 1:].values

# scaling values to be between 0 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
healthy_norm = scaler.fit_transform(healthy_train)

time_steps = 140
features = 1

# Reshape to fit them into the auto encoder. Second value is the timestep
healthy_reshape = healthy_norm.reshape(healthy_norm.shape[0], time_steps, features)

epochs = 5
batch_size = 70
heartbeat_to_plot = 6

create_model = False

if create_model:
    model = Sequential()
    history = createModel(epochs, batch_size, time_steps, features, model, healthy_reshape)
    modelnumber = get_current_modelnumber()
    model.save(f'model_number{modelnumber}.h5')
    document_model(modelnumber, model, epochs, batch_size, history)
    increase_modelnumber()
    plotReconstructed(heartbeat_to_plot, epochs, batch_size, history, model, modelnumber, scaler)
else:
    model = load_model("model_number1.h5")

print(calculate_mean(model, healthy_test))





