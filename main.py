
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import os


#  User input

ticker = input("Enter stock ticker (e.g., AAPL, TSLA, SST): ").upper()
years = int(input("Enter number of years of data (max 10): "))
years = min(years, 10)  # limit to 10 years

end_date = datetime.today()
start_date = end_date - timedelta(days=365*years)


# 2️⃣ Fetch data from yfinance

print(
    f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    raise ValueError("No data found for this ticker/date range!")

# Keep OHLC + Volume
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# Add technical indicators


def safe_divide(a, b):
    """Safe division to avoid division by zero"""
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


# Daily return (%)
df["Return"] = safe_divide(df["Close"].diff(), df["Close"].shift(1)).fillna(0)

# High-Low spread
df["HL_Spread"] = safe_divide(df["High"] - df["Low"], df["Close"])

# Moving averages with more robust filling
for window in [5, 10, 20]:
    df[f"MA{window}"] = df["Close"].rolling(window).mean()
# Fill NaN values using the new method
df = df.bfill().ffill()

# RSI with safer calculations
delta = df["Close"].diff().fillna(0)
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.rolling(14).mean().fillna(0)
avg_loss = loss.rolling(14).mean().fillna(1e-9)  # Avoid division by zero
rs = safe_divide(avg_gain, avg_loss)
df["RSI"] = 100 - (100 / (1 + rs))
df["RSI"] = df["RSI"].clip(0, 100).fillna(50)

# MACD with bounds
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = (ema12 - ema26).fillna(0)
df["MACD"] = df["MACD"].clip(-50, 50)

# Ensure no NaN values remain using the new method
df = df.bfill().ffill()


#  Prepare features and target

feature_cols = ["Open", "High", "Low", "Close", "Volume", "Return", "HL_Spread",
                "MA5", "MA10", "MA20", "RSI", "MACD"]
target_col = "Close"

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

# Data validation
print("Data summary before scaling:")
print(f"Features shape: {X.shape}")
print(f"NaN values in features: {np.isnan(X).sum()}")
print(f"Inf values in features: {np.isinf(X).sum()}")

# Remove any remaining NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

# Scale features and target separately
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

print(f"Features scaled range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
print(f"Target scaled range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]")


#  Create sequences

def create_sequences(X, y, seq_length=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)


seq_length = 60
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

print(f"Sequences created: {X_seq.shape}")

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32).squeeze()  # shape (N,)


# 6️⃣ Train / Val / Test split

train_size = int(len(X_tensor) * 0.7)
val_size = int(len(X_tensor) * 0.1)

X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]

X_val = X_tensor[train_size:train_size+val_size]
y_val = y_tensor[train_size:train_size+val_size]

X_test = X_tensor[train_size+val_size:]
y_test = y_tensor[train_size+val_size:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

train_loader = DataLoader(TensorDataset(
    X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


#  Define LSTM with weight initialization


class StockLSTM(nn.Module):
    def __init__(self, input_size=X_seq.shape[2], hidden_size=128, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # Apply dropout to last hidden state
        out = self.fc(out)
        return out.squeeze()


model = StockLSTM()

# Weight initialization function


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)


model.apply(init_weights)


# 8️⃣ Train model with improved early stopping

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

epochs = 200
best_val_loss = float('inf')
patience = 15
trigger_times = 0
model_saved = False

print("\nStarting training...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    batches_processed = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected in epoch {epoch+1}, skipping batch")
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        batches_processed += 1

    if batches_processed == 0:
        print(f"Epoch {epoch+1}: All batches had NaN loss, stopping training")
        break

    train_loss = total_loss / batches_processed

    # Validation
    model.eval()
    val_loss_total = 0
    val_batches = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss = criterion(val_outputs, y_val_batch)
            if not torch.isnan(val_loss):
                val_loss_total += val_loss.item()
                val_batches += 1

    if val_batches == 0:
        print(f"Epoch {epoch+1}: All validation batches had NaN loss")
        val_loss = float('inf')
    else:
        val_loss = val_loss_total / val_batches

    scheduler.step(val_loss)

    # Print learning rate every 10 epochs
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
    else:
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss and not np.isnan(val_loss):
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pth")
        model_saved = True
        print(f"New best model saved with val loss: {val_loss:.6f}")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

# Load best model if available
if model_saved and os.path.exists("best_model.pth"):
    print("Loading best model...")
    model.load_state_dict(torch.load("best_model.pth"))
else:
    print("No valid model saved, using current model")
    # Save current model if it's not completely broken
    if 'train_loss' in locals() and not np.isnan(train_loss):
        torch.save(model.state_dict(), "best_model.pth")
        model_saved = True


# Predict next day

if model_saved:
    model.eval()
    last_seq = torch.tensor(
        X_scaled[-seq_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(last_seq).item()

    pred_price = scaler_y.inverse_transform(
        np.array(pred_scaled).reshape(-1, 1))[0, 0]

    # FIX: Convert Series to scalar using .iloc[-1] and .item()
    current_price = df["Close"].iloc[-1]
    if hasattr(current_price, 'item'):  # If it's a Series or array
        current_price = current_price.item()
    current_price = float(current_price)  # Ensure it's a float

    change = pred_price - current_price
    change_pct = (change / current_price) * 100 if current_price != 0 else 0

    print(f"\nCurrent closing price: {current_price:.2f} USD")
    print(
        f"Predicted closing price for tomorrow ({ticker}): {pred_price:.2f} USD")
    print(f"Predicted change: {change:+.2f} USD ({change_pct:+.2f}%)")
else:
    print("\nCannot make prediction - no valid model available")


# Evaluate and plot (if we have a valid model)

if model_saved and len(X_test) > 0:
    y_test_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_pred_batch = model(X_batch)
            y_test_pred.extend(y_pred_batch.numpy())

    # Convert tensors to numpy arrays first
    y_test_np = y_test.numpy() if torch.is_tensor(y_test) else y_test
    y_test_actual_inv = scaler_y.inverse_transform(
        y_test_np.reshape(-1, 1)).flatten()
    y_test_pred_inv = scaler_y.inverse_transform(
        np.array(y_test_pred).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_actual_inv, y_test_pred_inv))
    mae = mean_absolute_error(y_test_actual_inv, y_test_pred_inv)

    # Safe MAPE calculation
    if np.all(y_test_actual_inv != 0):
        mape = np.mean(
            np.abs((y_test_actual_inv - y_test_pred_inv) / y_test_actual_inv)) * 100
    else:
        mape = float('inf')

    print(f"\nModel Performance on Test Set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    if mape != float('inf'):
        print(f"MAPE: {mape:.2f}%")
    else:
        print("MAPE: Cannot calculate (division by zero)")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual_inv, label="Actual Price", linewidth=2)
    plt.plot(y_test_pred_inv, label="Predicted Price", linewidth=2)
    plt.title(f"{ticker} Stock Price Prediction (Test Set)")
    plt.xlabel("Days")
    plt.ylabel("Price USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\nSkipping evaluation - no valid model or test data available")

print("\nTraining completed!")

"""**Note:** The timezone handling code in the previous cell was commented out due to persistent errors. If timezone awareness is critical for your analysis, you may need to investigate this further."""
