Neuromorphic AI for Microgrid Energy Optimization

This repository contains a Python implementation of a microgrid energy management system using a Spiking Neural Network (SNN) to optimize energy flows. The system integrates real household power consumption data from the UCI Individual Household Electric Power Consumption dataset and simulated renewable energy data (solar and wind). It models a battery storage system and minimizes energy costs by optimizing load allocation, battery charge/discharge, and grid import/export.

Features





Data Processing: Downloads and preprocesses household power consumption data and simulates renewable energy data.



Battery Model: Simulates a 5 kWh battery with 95% efficiency and state-of-charge constraints.



SNN Optimization: Uses a spiking neural network (via snntorch) to optimize energy flows based on load, renewable generation, battery state, and grid prices.



Cost Minimization: Incorporates grid prices to minimize energy costs.



Visualization: Generates plots showing load allocation, renewable generation, battery state, and grid interaction.

Prerequisites





Python: Version 3.8 or higher.



Dependencies:





pandas



numpy



requests



snntorch



torch



matplotlib



System Requirements:





At least 8GB RAM (due to the large UCI dataset).



Internet connection for downloading the dataset.



Optional: GPU for faster SNN training (PyTorch CUDA support).

Installation





Clone the Repository:

git clone (https://github.com/Jeevan-hub1/Neuromorphic-AI-for-Microgrid-Energy-Optimization)
cd neuromorphic-microgrid-optimization



Set Up a Virtual Environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Dependencies:

pip install pandas numpy requests snntorch torch matplotlib



Verify PyTorch Installation: Ensure PyTorch is version 2.0 or higher:

import torch
print(torch.__version__)

Usage





Run the Script:

python microgrid_optimization.py





The script downloads the UCI dataset (~144MB zip, ~2GB uncompressed) to a data folder.



It generates simulated renewable data and saves it as data/renewable_energy.csv.



It processes data, simulates a battery, trains the SNN, and optimizes energy flows.



Outputs include:





microgrid_optimization_advanced.png: A four-panel plot showing results.



predictions.csv: Predicted load, battery, and grid usage.



Console output with the total energy cost (e.g., $100.00–$1000.00).



Expected Runtime:





First run: ~20 minutes (due to dataset download and extraction).



Subsequent runs: ~5–10 minutes (depending on hardware and SNN training).



Training may stop early if loss plateaus (patience=10 epochs).



Output Files:





data/household_power_consumption.txt: Raw UCI dataset.



data/renewable_energy.csv: Simulated solar and wind data.



microgrid_optimization_advanced.png: Visualization of results.



predictions.csv: SNN predictions.

Project Structure

neuromorphic-microgrid-optimization/
├── data/                           # Folder for datasets
│   ├── household_power_consumption.txt  # UCI dataset
│   └── renewable_energy.csv        # Simulated renewable data
├── microgrid_optimization.py       # Main script
├── microgrid_optimization_advanced.png  # Output plot
├── predictions.csv                 # Output predictions
└── README.md                       # This file

How It Works





Data Download:





Downloads the UCI Household Power Consumption dataset (2006–2010, minute-level data).



Simulates hourly solar and wind data for 2006–2007 to align with UCI data.



Preprocessing:





Resamples data to hourly, aligns time indices, and normalizes to [0, 1].



Handles missing values and edge cases (e.g., NaN, identical values).



Battery Simulation:





Models a 5 kWh battery with 95% efficiency, charging from excess renewables and discharging to meet load.



SNN Optimization:





Uses a three-layer SNN with Leaky Integrate-and-Fire neurons and dropout (0.2).



Inputs: Normalized load, solar, wind, battery state, grid prices.



Outputs: Load allocation, battery charge/discharge, grid import/export.



Trains for up to 100 epochs with early stopping, minimizing MSE and grid costs.



Visualization:





Plots load allocation, renewable generation, battery state, and grid interaction.



Shows total energy cost in the grid panel title.

Troubleshooting





Load Data Issues:





If Load: [0.0000, 0.0000] appears, the UCI dataset may be corrupted. Delete data/ and re-run.



Check data/household_power_consumption.txt for missing Global_active_power values.



Shape Errors:





Ensure tensor shapes match (printed during training). If not, verify snntorch and torch versions.



Memory Errors:





The UCI dataset is large. Close other applications or use a system with more RAM.



Dependency Issues:





Update dependencies: pip install --upgrade snntorch torch pandas numpy matplotlib requests.



Verify PyTorch: python -c "import torch; print(torch.__version__)".

Future Improvements





Replace simulated renewable data with real NREL data (e.g., via NREL API).



Add train/test split for SNN validation.



Tune SNN hyperparameters (e.g., hidden layer sizes, learning rate).



Parallelize training with mini-batches using torch.DataLoader.



Model battery degradation or temperature effects.

Contributing

Contributions are welcome! Please:





Fork the repository.



Create a feature branch (git checkout -b feature-name).



Commit changes (git commit -m "Add feature").



Push to the branch (git push origin feature-name).



Open a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments





UCI Machine Learning Repository for the household power consumption dataset.



snntorch for spiking neural network support.



PyTorch for deep learning framework.

Contact

For questions or issues, please open an issue on GitHub or contact nandakumarponnala@gmail.com.

