# Neuromorphic AI for Microgrid Energy Optimization

A cutting-edge Python implementation of a smart microgrid energy management system, powered by **Spiking Neural Networks (SNNs)**. This project uses **neuromorphic AI** to intelligently balance energy flows in a microgrid environment using rAeal energy consumption data and simulated renewable sources.

---

## 🔍 Overview

This system integrates:

* Real household data from the **UCI Power Consumption Dataset**
* Simulated **solar** and **wind** renewable energy inputs
* A dynamic **battery storage model**
* A **Spiking Neural Network (SNN)** using `snntorch` to optimize energy allocation

The objective is to **minimize energy costs** while ensuring stable power delivery, balancing demand, battery storage, and grid imports/exports.

---

## 🚀 Features

* **📊 Data Processing**: Cleans, resamples, and aligns real-world and simulated data
* **🔋 Battery Model**: Simulates a 5 kWh battery with 95% round-trip efficiency
* **🧠 SNN Optimization**: Trains an SNN to optimize load vs generation decisions
* **💸 Cost Minimization**: Integrates dynamic grid prices into the optimization
* **📈 Visualization**: Plots showing energy flows, battery state, and cost savings

---

## 🛠️ Installation

```bash
git clone https://github.com/Jeevan-hub1/Neuromorphic-AI-for-Microgrid-Energy-Optimization.git
cd neuromorphic-microgrid-optimization

# Optional: Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### ✅ Dependencies

* `pandas`, `numpy`, `matplotlib`
* `snntorch`, `torch`, `requests`

Ensure PyTorch is installed with:

```python
import torch
print(torch.__version__)  # Should be >= 2.0
```

---

## 📦 Usage

Run the full pipeline with:

```bash
python microgrid_optimization.py
```

The script:

* Downloads the UCI dataset (\~144MB zip, \~2GB uncompressed)
* Simulates 1 year of solar & wind data
* Trains an SNN for energy optimization
* Outputs:

  * `microgrid_optimization_advanced.png`
  * `predictions.csv`

Expected Runtime:

* **First run**: \~20 mins (includes data download & training)
* **Subsequent runs**: \~5–10 mins

---

## 📂 Project Structure

```
neuromorphic-microgrid-optimization/
├── data/
│   ├── household_power_consumption.txt
│   └── renewable_energy.csv
├── microgrid_optimization.py
├── microgrid_optimization_advanced.png
├── predictions.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 🔻 Data

* Real data from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
* Simulated renewable energy (solar & wind)

### 🧼 Preprocessing

* Hourly resampling
* Normalization
* Handling missing values

### 🔋 Battery Model

* 5 kWh capacity
* 95% round-trip efficiency
* Constraints on max charge/discharge

### 🔁 SNN Model (via `snntorch`)

* Inputs: Normalized load, solar, wind, battery state, grid prices
* Outputs: Battery dispatch, grid import/export, load served
* Optimized with MSE + cost penalty loss function

### 📊 Visualization

* 4-panel plot:

  * Load
  * Solar/Wind Generation
  * Battery SOC
  * Grid Usage & Cost

---

## ⚠️ Troubleshooting

* **Corrupted Load Data**: Re-delete the `data/` folder and rerun
* **Tensor Shape Errors**: Ensure snntorch and torch versions match
* **Memory Issues**: Close apps or reduce dataset duration

---

## 💡 Future Work

* Integrate real solar/wind data via NREL API
* Add test/train split and validation metrics
* Model battery degradation and weather impact
* Use DataLoader for batch SNN training

---

## 📬 Contact

**Author**: Jeevan Reddy Ponnala
For questions: [nandakumarponnala@gmail.com](mailto:nandakumarponnala@gmail.com)
GitHub: [github.com/Jeevan-hub1](https://github.com/Jeevan-hub1)

---



---

## 🙏 Acknowledgments

* [UCI ML Repository](https://archive.ics.uci.edu)
* [snntorch](https://snntorch.readthedocs.io)
* [PyTorch](https://pytorch.org)
