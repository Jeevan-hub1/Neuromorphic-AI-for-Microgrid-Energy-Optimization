# Neuromorphic AI for Microgrid Energy Optimization

This repository implements a novel energy optimization system for small-scale microgrids using **Neuromorphic AI** (Spiking Neural Networks) and lightweight **Machine Learning** (Graph Neural Networks, Tiny Transformers). The system optimizes energy distribution, detects anomalies, and forecasts demand/generation using a **small, effective dataset**. Designed for advanced users, it targets efficient, low-power deployment on edge devices.

## Features
- **Spiking Neural Networks (SNNs)**: Event-driven anomaly detection for real-time energy data.
- **Graph Neural Networks (GNNs)**: Lightweight, quantized model for short-term energy demand forecasting.
- **Tiny Transformer**: Compact model for predicting renewable energy generation (e.g., solar).
- **Reinforcement Learning (RL)**: Optimizes energy allocation for efficiency.
- **Small Dataset**: ~20,000 data points (1 month of hourly microgrid data).
- **Simulation**: Microgrid simulation using GridLAB-D for testing.

## Dataset
- **Source**: Synthetic microgrid dataset (`data/microgrid_data.csv`) with energy consumption, solar/wind generation, and weather data.
- **Size**: ~20,000 rows (hourly data for 1 month, 1 solar panel, 10 loads).
- **Features**:
  - `timestamp`: Date and time.
  - `consumption_kwh`: Energy demand per load.
  - `solar_kwh`: Solar generation.
  - `wind_kwh`: Wind generation.
  - `temperature_c`: Weather data.
  - `battery_soc`: Battery state of charge.
- **Generation**: Use `data/generate_dataset.py` to create custom synthetic data.

## Requirements
- Python 3.8+
- PyTorch, PyTorch Geometric, Brian2, Stable-Baselines3, NumPy, Pandas
- GridLAB-D (for simulation)
- See `requirements.txt` for full dependencies.

## Installation
```bash
git clone https://github.com/<your-username>/Neuromorphic-Microgrid-Optimization.git
cd Neuromorphic-Microgrid-Optimization
pip install -r requirements.txt
