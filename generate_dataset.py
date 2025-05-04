import pandas as pd
import numpy as np
import requests
import os
from snntorch import spikegen
import snntorch
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Safe normalization function
def safe_normalize(data):
    """
    Normalizes data to [0, 1], handling edge cases where max == min or NaN/infs.
    """
    data = np.array(data, dtype=np.float32)
    if np.all(np.isnan(data)) or np.all(np.isinf(data)) or len(data) == 0:
        return np.zeros_like(data) if len(data) > 0 else np.array([])
    if np.max(data) == np.min(data) or np.isnan(np.max(data)) or np.isnan(np.min(data)):
        return np.zeros_like(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0, 1)

# Step 1: Download real-life microgrid and renewable data
def download_data():
    """
    Downloads household power consumption and simulates renewable energy data.
    Uses UCI Household dataset for load and a CSV for renewable data (simulated).
    """
    data_dir = "data"
    load_path = os.path.join(data_dir, "household_power_consumption.txt")
    renewable_path = os.path.join(data_dir, "renewable_energy.csv")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download household load data
    if not os.path.exists(load_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
        print("Downloading load dataset...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            zip_path = os.path.join(data_dir, "household_power_consumption.zip")
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Load dataset downloaded and extracted.")
        except requests.RequestException as e:
            print(f"Failed to download load dataset: {e}")
            raise

    # Simulate renewable data (aligned with UCI dataset timeline)
    if not os.path.exists(renewable_path):
        print("Simulating renewable energy data (replace with real NREL data for production)...")
        hours = 8760  # One year of hourly data
        time_index = pd.date_range(start="2006-12-16", periods=hours, freq='h')  # UCI dataset start
        solar = np.abs(np.sin(np.linspace(0, 2 * np.pi * hours / 24, hours))) * 2.0 + np.random.normal(0, 0.1, hours)
        wind = np.abs(np.random.normal(0.5, 0.3, hours)) * 1.5
        df_renewable = pd.DataFrame({
            'datetime': time_index,
            'solar_power': np.clip(solar, 0, None),
            'wind_power': np.clip(wind, 0, None)
        })
        df_renewable.to_csv(renewable_path, index=False)
        print("Renewable dataset created.")

    return load_path, renewable_path

# Step 2: Preprocess load and renewable data
def preprocess_data(load_path, renewable_path):
    """
    Loads and preprocesses household and renewable data. Aligns time indices and normalizes.
    """
    # Load household data
    df_load = pd.read_csv(load_path, sep=';', na_values='?', low_memory=False)
    df_load['datetime'] = pd.to_datetime(df_load['Date'] + ' ' + df_load['Time'], format='%d/%m/%Y %H:%M:%S')
    df_load.set_index('datetime', inplace=True)
    df_load = df_load[['Global_active_power']].dropna()
    df_load_hourly = df_load.resample('h').mean().fillna(method='ffill')  # Fill NaNs

    # Load renewable data
    df_renewable = pd.read_csv(renewable_path)
    df_renewable['datetime'] = pd.to_datetime(df_renewable['datetime'])
    df_renewable.set_index('datetime', inplace=True)
    df_renewable_hourly = df_renewable.resample('h').mean().fillna(method='ffill')  # Fill NaNs

    # Align time indices
    common_index = df_load_hourly.index.intersection(df_renewable_hourly.index)
    if len(common_index) == 0:
        raise ValueError("No common time index between load and renewable data.")
    print(f"Common index size: {len(common_index)}")
    df_load_hourly = df_load_hourly.loc[common_index]
    df_renewable_hourly = df_renewable_hourly.loc[common_index]

    # Extract data
    load_data = df_load_hourly['Global_active_power'].values
    solar_data = df_renewable_hourly['solar_power'].values
    wind_data = df_renewable_hourly['wind_power'].values

    # Validate load data
    if len(load_data) == 0 or np.all(np.isnan(load_data)):
        raise ValueError("Load data is empty or all NaN after preprocessing.")
    print(f"Load data range before normalization: [{np.min(load_data):.4f}, {np.max(load_data):.4f}]")

    # Normalize data
    load_normalized = safe_normalize(load_data)
    solar_normalized = safe_normalize(solar_data)
    wind_normalized = safe_normalize(wind_data)

    return load_normalized, solar_normalized, wind_normalized, common_index

# Step 3: Simulate advanced battery model
def simulate_battery_model(load_data, solar_data, wind_data, time_index):
    """
    Simulates battery storage with efficiency losses and SoC constraints.
    """
    hours = len(time_index)
    battery_capacity = 5.0  # kWh
    battery_state = np.zeros(hours)
    battery_state[0] = battery_capacity * 0.5
    efficiency = 0.95  # Round-trip efficiency

    for t in range(1, hours):
        renewable_gen = solar_data[t] + wind_data[t]
        net_load = load_data[t] - renewable_gen

        if net_load > 0:  # Discharge
            discharge = min(net_load, battery_state[t-1] * efficiency, 0.5)
            battery_state[t] = battery_state[t-1] - discharge / efficiency
        else:  # Charge
            charge = min(-net_load * efficiency, (battery_capacity - battery_state[t-1]) * efficiency, 0.5)
            battery_state[t] = battery_state[t-1] + charge / efficiency

        battery_state[t] = np.clip(battery_state[t], 0, battery_capacity)

    return battery_state

# Step 4: Define enhanced SNN with dropout
class NeuromorphicEnergyOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuromorphicEnergyOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.lif1 = snntorch.Leaky(beta=0.9)
        self.lif2 = snntorch.Leaky(beta=0.85)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_out = []
        
        # x shape: [num_steps, num_samples, input_size]
        for t in range(x.shape[0]):
            cur1 = self.fc1(x[t])  # [num_samples, hidden_size1]
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.dropout1(spk1)
            cur2 = self.fc2(spk1)  # [num_samples, hidden_size2]
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.dropout2(spk2)
            cur3 = self.fc3(spk2)  # [num_samples, output_size]
            spk_out.append(cur3)
        
        return torch.stack(spk_out)  # [num_steps, num_samples, output_size]

# Step 5: Simulate microgrid optimization with cost
def simulate_microgrid_optimization(load_data, solar_data, wind_data, battery_state, time_index):
    """
    Simulates energy optimization with cost minimization using SNN.
    """
    # Simulate grid prices (simple diurnal pattern)
    hours = len(time_index)
    grid_prices = 0.1 + 0.05 * np.sin(np.linspace(0, 2 * np.pi * hours / 24, hours))

    # Normalize all inputs
    load_normalized = safe_normalize(load_data)
    solar_normalized = safe_normalize(solar_data)
    wind_normalized = safe_normalize(wind_data)
    battery_normalized = safe_normalize(battery_state)
    grid_prices_normalized = safe_normalize(grid_prices)

    # Debug: Print ranges to verify normalization
    print(f"Input ranges after normalization:")
    print(f"Load: [{np.min(load_normalized):.4f}, {np.max(load_normalized):.4f}]")
    print(f"Solar: [{np.min(solar_normalized):.4f}, {np.max(solar_normalized):.4f}]")
    print(f"Wind: [{np.min(wind_normalized):.4f}, {np.max(wind_normalized):.4f}]")
    print(f"Battery: [{np.min(battery_normalized):.4f}, {np.max(battery_normalized):.4f}]")
    print(f"Grid Prices: [{np.min(grid_prices_normalized):.4f}, {np.max(grid_prices_normalized):.4f}]")

    # Combine inputs
    input_data = np.stack([load_normalized, solar_normalized, wind_normalized, battery_normalized, grid_prices_normalized], axis=1)

    # Check for NaN/infinite values
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        print("Warning: NaN or infinite values detected in input_data. Replacing with zeros.")
        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to spikes
    spike_data = spikegen.rate(torch.tensor(input_data, dtype=torch.float32), num_steps=100)  # [num_steps, num_samples, input_size]

    # Initialize SNN
    input_size = 5  # Load, solar, wind, battery, grid price
    hidden_size1 = 20
    hidden_size2 = 10
    output_size = 3  # Load allocation, battery charge/discharge, grid import/export
    model = NeuromorphicEnergyOptimizer(input_size, hidden_size1, hidden_size2, output_size)

    # Custom loss function incorporating grid cost
    def cost_loss(output, target, grid_prices):
        # output: [num_steps, num_samples, output_size], target: [num_samples, output_size]
        mse_loss = nn.MSELoss()(output[:, :, :2], target[None, :, :2].expand(output.shape[0], -1, -1))  # Load and battery
        grid_cost = torch.sum(output[:, :, 2] * torch.tensor(grid_prices[:output.shape[1]], dtype=torch.float32))
        return mse_loss + 0.01 * grid_cost  # Weight cost term

    # Training loop with early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    target = torch.tensor(np.stack([load_normalized, battery_normalized, np.zeros_like(load_normalized)], axis=1), dtype=torch.float32)
    
    print(f"Spike data shape: {spike_data.shape}")
    print(f"Target shape: {target.shape}")

    model.train()
    best_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(100):  # Increased epochs
        optimizer.zero_grad()
        output = model(spike_data)  # [num_steps, num_samples, output_size]
        print(f"Output shape: {output.shape}")
        loss = cost_loss(output, target, grid_prices_normalized)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(spike_data).mean(dim=0).numpy()  # Average over time steps: [num_samples, output_size]

    # Save predictions
    df_predictions = pd.DataFrame({
        'datetime': time_index[:len(predictions)],
        'predicted_load': predictions[:, 0],
        'predicted_battery': predictions[:, 1],
        'grid_usage': predictions[:, 2]
    })
    df_predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to 'predictions.csv'.")

    # Calculate costs (using original grid prices)
    grid_usage = predictions[:, 2]  # Grid import/export
    energy_cost = np.sum(grid_usage * grid_prices[:len(grid_usage)])

    # Plot results
    plt.figure(figsize=(14, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_index[:len(load_normalized)], load_normalized, label="Actual Load")
    plt.plot(time_index[:len(predictions)], predictions[:, 0], label="Predicted Load Allocation")
    plt.xlabel("Time")
    plt.ylabel("Normalized Power")
    plt.title("Load Allocation")
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(time_index[:len(solar_normalized)], solar_normalized, label="Solar Generation")
    plt.plot(time_index[:len(wind_normalized)], wind_normalized, label="Wind Generation")
    plt.xlabel("Time")
    plt.ylabel("Normalized Power")
    plt.title("Renewable Energy Generation")
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(time_index[:len(battery_normalized)], battery_normalized, label="Battery State")
    plt.plot(time_index[:len(predictions)], predictions[:, 1], label="Predicted Battery Charge/Discharge")
    plt.xlabel("Time")
    plt.ylabel("Normalized Battery State")
    plt.title("Battery Storage Optimization")
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(time_index[:len(grid_usage)], grid_usage, label="Grid Import/Export")
    plt.plot(time_index[:len(grid_prices)], grid_prices[:len(grid_usage)], label="Grid Price")
    plt.xlabel("Time")
    plt.ylabel("Power/Price")
    plt.title(f"Grid Interaction (Total Cost: ${energy_cost:.2f})")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("microgrid_optimization_advanced.png")
    plt.close()

    return energy_cost

# Main execution
if __name__ == "__main__":
    # Download and preprocess data
    load_path, renewable_path = download_data()
    load_data, solar_data, wind_data, time_index = preprocess_data(load_path, renewable_path)
    
    # Simulate battery
    battery_state = simulate_battery_model(load_data, solar_data, wind_data, time_index)
    
    # Run simulation
    energy_cost = simulate_microgrid_optimization(load_data, solar_data, wind_data, battery_state, time_index)
    print(f"Simulation complete. Total energy cost: ${energy_cost:.2f}")
    print("Check 'microgrid_optimization_advanced.png' for results and 'predictions.csv' for predictions.")