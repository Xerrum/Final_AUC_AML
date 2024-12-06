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
import seaborn as sns

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
    If the file doesn't exist, it creates one with an initial value of 1.
    :return: ---
    """
    try:
        with open("modelnumber.txt", 'r') as f:
            value = int(f.read())
    except FileNotFoundError:
        value = 0
    value += 1
    with open("modelnumber.txt", 'w') as f:
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


def calculate_mean_and_save(model, test_data, scaler, file_path, threshold):
    """
    Calculates the mean, max, and min MSE values for the model predicting the test_data and saves the losses to a file.

    :param model: The trained model for predictions.
    :param test_data: The test data to evaluate.
    :param scaler: The scaler used for preprocessing the data.
    :param file_path: Path to save the losses (default: 'losses.json').
    :param threshold: loss threshold that decides if heartbeat is healthy or not
    :return: mean loss, max loss, and min loss
    """
    # Scale the test data
    test_data_scaled = scaler.fit_transform(test_data)

    # Compute losses
    losses = []
    healthy = 0
    non_healthy = 0

    for i in test_data_scaled:
        input_heartbeat = i.reshape(1, 140, 1)  # Assuming 140 is the length of the input data
        reconstructed_heartbeat = model.predict(input_heartbeat)
        mse = mean_squared_error(reconstructed_heartbeat.reshape(-1), input_heartbeat.reshape(-1))
        losses.append(mse)
        if mse > threshold:
            non_healthy += 1
        else:
            healthy += 1

    # # Convert losses to numpy array
    # losses_np = np.array(losses)
    #
    # # Calculate mean, max, and min losses
    # mean_loss = np.mean(losses_np)
    # max_loss = np.max(losses_np)
    # min_loss = np.min(losses_np)

    # # Save losses to a file in JSON format
    # with open(file_path, "w") as file:
    #    json.dump({"mean_loss": mean_loss, "max_loss": max_loss, "min_loss": min_loss, "all_losses": losses}, file)
#    mean_loss, max_loss, min_loss ADD THIS LINE INTO THE RETURN TO RETURN MEAN AND MAX LOSS
    return healthy, non_healthy


def calculate_confusion_matrix(data):
    """
    Calculate the confusion matrix and metrics based on the input data.

    Parameters:
        data (list of lists): Each sublist contains two values:
                              [classified_as_healthy, classified_as_non_healthy].
                              The first sublist corresponds to healthy heartbeats,
                              and the others correspond to unhealthy heartbeats.

    Returns:
        confusion_matrix (np.ndarray): The calculated confusion matrix as a 2x2 array.
        metrics (dict): A dictionary with accuracy, precision, recall, and F1-score.
    """
    # Calculate True Positives (TP) and False Negatives (FN) from healthy heartbeats
    TP, FN = data[0]

    # Calculate False Positives (FP) and True Negatives (TN) from unhealthy heartbeats
    FP = sum(row[0] for row in data[1:])  # Sum of "classified_as_healthy" for all unhealthy types
    TN = sum(row[1] for row in data[1:])  # Sum of "classified_as_non_healthy" for all unhealthy types

    # Confusion matrix
    confusion_matrix = np.array([[TP, FN], [FP, TN]])

    # Metrics
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return confusion_matrix, metrics


def plot_confusion_matrix(confusion_matrix):
    """
    Plot the confusion matrix using a heatmap.

    Parameters:
        confusion_matrix (np.ndarray): A 2x2 confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Healthy", "Predicted Non-Healthy"],
                yticklabels=["Actual Healthy", "Actual Non-Healthy"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig("confusion_matrix.png", dpi=300)  # High-resolution for publications
    plt.show()

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

# scaling values to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
healthy_norm = scaler.fit_transform(healthy_train)

# time steps is the window size that's being used for the model to train on the heartbeat
time_steps = 140
features = 1

# Reshape to fit them into the auto encoder. Second value is the timestep
healthy_reshape = healthy_norm.reshape(healthy_norm.shape[0], time_steps, features)

create_model = False

# creates model with settings described below
if create_model:
    epochs = 5
    batch_size = 70
    heartbeat_to_plot = 6
    model = Sequential()
    history = createModel(epochs, batch_size, time_steps, features, model, healthy_reshape)
    modelnumber = get_current_modelnumber()
    model.save(f'model_number{modelnumber}.h5')
    document_model(modelnumber, model, epochs, batch_size, history)
    increase_modelnumber()
    plotReconstructed(heartbeat_to_plot, epochs, batch_size, history, model, modelnumber, scaler)
else:
    model = load_model("model_number1.h5")

# calculates mean, max and min value of a model which is specified below on the test set
calculate_mean = True

threshold = 0.075
labels = []

if calculate_mean:
    label_healthy, label_unhealthy = (calculate_mean_and_save(model, healthy_test, scaler, f"{threshold}threshold_healthy.json", threshold))
    labels.append([label_healthy, label_unhealthy])
    losses_names = [f"{threshold}threshold_arrhythmia.json", f"{threshold}threshold_prematureV.json", f"{threshold}threshold_prematureA.json", f"{threshold}threshold_other.json"]
    j = 0
    for i in [arrhythmia_df, prematureV_df, prematureA_df, other_df]:
        label_healthy, label_unhealthy = calculate_mean_and_save(model, i.iloc[:, 1:].values, scaler, losses_names[j], threshold)
        labels.append([label_healthy, label_unhealthy])
        j += 1

conf_matrix, metrics = calculate_confusion_matrix(labels)