import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np


def setupdf(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
    df.rename(columns={0: 'label'}, inplace=True)
    for i in range(1, len(df.columns)):
        df.rename(columns={i: i}, inplace=True)
    # print(df)
    return df


def setupdfs(df):
    healthy_df = df[df['label'] == 1.0]
    arrhythmia_df = df[df['label'] == 2.0]
    prematureV_df = df[df['label'] == 3.0]
    prematureA_df = df[df['label'] == 4.0]
    other_df = df[df['label'] == 5.0]
    return [healthy_df, arrhythmia_df, prematureV_df, prematureA_df, other_df]


def plot_df(dataframe, num, title):
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
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data_array[1])), data_array[5], label=f'Heartbeat {num + 1}')
    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Pulse Value')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    plt.show()


def plot_eachtype(dataframes):
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


train_df = setupdf("ECG5000_TRAIN.txt")
test_df = setupdf("ECG5000_TEST.txt")

# combining train and test dataframe
combineddf = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# splitting according to the different labels
[healthy_df, arrhythmia_df, prematureV_df, prematureA_df, other_df] = setupdfs(combineddf)
print(healthy_df.shape, arrhythmia_df.shape, prematureV_df.shape, prematureA_df.shape, other_df.shape)
# shapes: 2919,141 1767,141 96, 141 194,141 24,141

# reshaping for further use, after it will be a 2D matrix. Take only 583 from 2919 heartbeats to maintain 20%-80% Train-Test-Split
healthy = healthy_df.iloc[:, 1:].values

model = Sequential()

# scaling values to be between 0 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
healthy_norm = scaler.fit_transform(healthy)

# Reshape to fit them into the auto encoder. Second value is the timestep
healthy_reshape = healthy.reshape((healthy.shape[0], 140, 1))

#plot_df(train_df, 1, 'Arrhythmia Heartbeat Over Time')
#plot_eachtype([healthy_df, arrhythmia_df, prematureV_df, prematureA_df, other_df])

def createModel():
    epochs = 100
    batch_size = 32

    # Encoder
    model.add(LSTM(64, activation='relu', input_shape=(140, 1), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))

    # Bottleneck
    model.add(Dense(10, activation='relu'))  # Smallest representation layer (bottleneck)

    # Decoder
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(140, activation='sigmoid'))  # Output layer matches the input feature size for each timestep

    # 5. Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(healthy_reshape, healthy_reshape, epochs, batch_size, validation_split=0.1)


def plotReconstructed(num_heartbeat):
    # Select a single heartbeat from the input data
    input_heartbeat = healthy_reshape[num_heartbeat]  # For example, the fifth heartbeat

    # Get the reconstructed heartbeat from the model
    reconstructed_heartbeat = model.predict(np.expand_dims(input_heartbeat, axis=0)).reshape(-1)

    # Plotting both the input and reconstructed heartbeat
    plt.figure(figsize=(10, 5))
    plt.plot(input_heartbeat.reshape(-1), label="Input Heartbeat", linestyle='--', color='blue')
    plt.plot(reconstructed_heartbeat, label="Reconstructed Heartbeat", linestyle='-', color='red')
    plt.title("Comparison of Input and Reconstructed Heartbeat")
    plt.xlabel("Timestep")
    plt.ylabel("Pulse Value")
    plt.legend()
    plt.show()

createModel()

plotReconstructed(5)

plotNumGraph(healthy_norm, 5, "scaled healthy heartbeat num 5")