import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from momentfm import MOMENTPipeline

import torch.optim as optim
from torch import nn
from torchmetrics import MeanSquaredError, MeanAbsoluteError

# Load data
df = pd.read_csv("../data/preprocessed_hourly_data.csv", parse_dates=['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# Select features and target
features = ['Open', 'High', 'Low', 'Close', 'SMA_200', 'ATR_168']
target = 'Close'
data = df[features].values.astype(np.float32)

print(df.shape)
df.head()

# Normalize features channel-wise (fit on train portion only)
n_total = len(data)
split_idx = int(0.8 * n_total)

# Create sliding windows
input_len = 512
horizon = 360  # forecast horizon, 360 hours = 15 days

windows, targets = [], []
for i in range(n_total - input_len - horizon + 1):
    inp = data[i:i+input_len, :].T  # shape: (channels, 512)
    tgt = data[i+input_len:i+input_len+horizon, features.index(target)]  # Close prices
    windows.append(inp)
    targets.append(tgt)
print(inp.shape, tgt.shape)

windows = np.stack(windows)   # shape: (N_windows, channels, input_len=512)
targets = np.stack(targets)   # shape: (N_windows, horizon=360)

print(windows.shape, targets.shape)

# Split into train/val/test (chronological)
n_windows = len(windows)
print(f"Total number of windows: {n_windows}")

train_end = int(0.8 * n_windows)
val_end = int(0.9 * n_windows)
train_X, val_X, test_X = windows[:train_end], windows[train_end:val_end], windows[val_end:]
train_y, val_y, test_y = targets[:train_end], targets[train_end:val_end], targets[val_end:]

class ForecastDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()  # shape (N, C, L): (N_windows, channels=6, input_len=512)
        self.Y = torch.from_numpy(Y).float()  # shape (N, H): (N_windows, horizon=360)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        timeseries = self.X[idx]              # (C, L)
        forecast = self.Y[idx]               # (H,)
        input_mask = torch.ones(timeseries.shape[-1])  # (L,), all observed
        return timeseries, forecast, input_mask

train_ds = ForecastDataset(train_X, train_y)
val_ds   = ForecastDataset(val_X,   val_y)
test_ds  = ForecastDataset(test_X,  test_y)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

# Load MOMENT pipeline for forecasting
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={
        'task_name': 'forecasting',
        'forecast_horizon': horizon,
        'head_dropout': 0.1,
        'weight_decay': 0.0,
        'freeze_encoder': True,
        'freeze_embedder': True,
        'freeze_head': False
    }
)
model.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5 

# Loss functions
mse_loss_fn = nn.MSELoss()
huber_loss_fn = nn.SmoothL1Loss()
mse_loss_fn.to(device)
huber_loss_fn.to(device)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

# Metrics
metric_mse = MeanSquaredError().to(device)
metric_mae = MeanAbsoluteError().to(device)

# Scheduler
from torch.optim.lr_scheduler import OneCycleLR
total_steps = len(train_loader) * num_epochs
scheduler = OneCycleLR(optimizer, max_lr=1e-4, total_steps=total_steps)
max_grad_norm = 5.0

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for series, target, mask in train_loader:
        series = series.to(device)     # (batch, channels, L)
        target = target.to(device)     # (batch, H)
        mask   = mask.to(device)       # (batch, L)

        optimizer.zero_grad()
        output = model(x_enc=series, input_mask=mask)  # forward pass
        
        # Debug the output shape
        print(f"Output type: {type(output)}")
        if hasattr(output, 'forecast'):
            print(f"Output forecast shape: {output.forecast.shape}")
            # Check if we need to select a specific channel/feature
            if len(output.forecast.shape) == 3:  # [batch, channels, horizon]
                # Select the channel corresponding to 'Close' price (should be channel index 3)
                forecast = output.forecast[:, features.index(target), :]
                print(f"Selected forecast shape: {forecast.shape}")
                loss = mse_loss_fn(forecast, target)
            else:
                loss = mse_loss_fn(output.forecast, target)
        else:
            # If output is the forecast tensor directly
            print(f"Direct output shape: {output.shape}")
            if len(output.shape) == 3:  # [batch, channels, horizon]
                # Select the channel corresponding to 'Close' price
                forecast = output[:, features.index(target), :]
                print(f"Selected forecast shape: {forecast.shape}")
                loss = mse_loss_fn(forecast, target)
            else:
                loss = mse_loss_fn(output, target)
                
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)

    model.eval()
    val_losses = []
    metric_mse.reset(); metric_mae.reset()
    with torch.no_grad():
        for series, target, mask in val_loader:
            series = series.to(device)
            target = target.to(device)
            mask   = mask.to(device)

            output = model(x_enc=series, input_mask=mask)
            
            # Handle output shape like in training
            if hasattr(output, 'forecast'):
                if len(output.forecast.shape) == 3:  # [batch, channels, horizon]
                    forecast = output.forecast[:, features.index(target), :]
                else:
                    forecast = output.forecast
            else:
                if len(output.shape) == 3:  # [batch, channels, horizon]
                    forecast = output[:, features.index(target), :]
                else:
                    forecast = output
                    
            val_loss = mse_loss_fn(forecast, target)
            val_losses.append(val_loss.item())
            metric_mse(forecast, target)
            metric_mae(forecast, target)
            
    avg_val_loss = np.mean(val_losses)
    val_mse = metric_mse.compute().item()
    val_mae = metric_mae.compute().item()

    print(f"Epoch {epoch+1} | Train MSE {avg_train_loss:.4f} | Val MSE {val_mse:.4f}, Val MAE {val_mae:.4f}")

# Testing
model.eval()
test_losses = []
metric_mse.reset(); metric_mae.reset()
with torch.no_grad():
    for series, target, mask in DataLoader(test_ds, batch_size=16):
        series = series.to(device)
        target = target.to(device)
        mask   = mask.to(device)
        
        output = model(x_enc=series, input_mask=mask)
        
        # Handle output shape like in training
        if hasattr(output, 'forecast'):
            if len(output.forecast.shape) == 3:  # [batch, channels, horizon]
                forecast = output.forecast[:, features.index(target), :]
            else:
                forecast = output.forecast
        else:
            if len(output.shape) == 3:  # [batch, channels, horizon]
                forecast = output[:, features.index(target), :]
            else:
                forecast = output
                
        metric_mse(forecast, target)
        metric_mae(forecast, target)
        test_losses.append(huber_loss_fn(forecast, target).item())
        
test_mse = metric_mse.compute().item()
test_mae = metric_mae.compute().item()
test_huber = np.mean(test_losses)

print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test Huber: {test_huber:.4f}")