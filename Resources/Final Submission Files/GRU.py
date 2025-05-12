# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

# %%
# Load the dataset
df = pd.read_csv('preprocessed_hourly_data.csv')

# Check the first few rows
print(df.head())

# %%
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# %%
print(df.columns.tolist())

# %%
scaler = MinMaxScaler(feature_range=(0, 1))

# Select the columns you want to scale (for example, open, high, low, close, volume)
scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'SMA_200', 'ATR_168']])

# Convert the scaled data back into a DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=['Open', 'High', 'Low', 'Close', 'SMA_200', 'ATR_168'], index=df.index)

# %%
def create_sequences(data, time_steps=60):
    X = []
    y = []

    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])  # Sequence of 60 previous minutes
        y.append(data[i, 3])  # Target is the 'close' price

    return np.array(X), np.array(y)

# Create sequences
X, y = create_sequences(scaled_data, time_steps=60)

# Split data into training and testing sets (80% training, 20% testing)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%
model = Sequential()

# First GRU layer
model.add(GRU(units=100, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

# Second GRU layer (optional)
model.add(GRU(units=100, activation='tanh', return_sequences=False))
model.add(Dropout(0.3))

# Output layer
model.add(Dense(units=1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# %%
# Summary of the model
model.summary()

# %%
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# %%
# Evaluate the model on the test data
test_loss, test_mae = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# %%
# Make predictions
predicted_prices = model.predict(X_test)

# %%
# Ensure y_test_actual contains the actual 'Close' prices for the test set
y_test_actual = y_test  # 'y_test' already represents the 'Close' prices, so no need to reshape

# Reshape predicted prices for inverse transformation
predicted_prices = predicted_prices.reshape(-1, 1)

# Now inverse transform only the 'Close' prices
predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((predicted_prices.shape[0], 5)), predicted_prices), axis=1))[:, 5]
y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((y_test_actual.shape[0], 5)), y_test_actual.reshape(-1, 1)), axis=1))[:, 5]


# %%
# Plot actual vs predicted prices
plt.figure(figsize=(14, 7))
plt.plot(y_test_actual, label="Actual Prices")
plt.plot(predicted_prices, label="Predicted Prices")
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%
# Plot Training and Validation Loss curves
plt.figure(figsize=(10, 6))

# Training Loss
plt.plot(history.history['loss'], label='Training Loss', color='blue')

# Validation Loss
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')

# Adding title and labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Add legend
plt.legend()

# Show the plot
plt.show()

# %%



