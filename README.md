# Thesis-Project

This Repo will contain any coding work related to my master thesis.
It will include codes for trial and learning process even if it will not be part of the final code.
Additionally, it will contain the final project code.

The main final code files are the ones that build and train the models. There are three different models as followes:
- LSTM model for CPU timeseries forecasting (v4_CSC_data_lstm.py)
- Regression model for power modeling (v5_CSC_cpu_net_pow.py). CPU usage and network readings are used as input features and power radings is the target.
- Regression model for temperature modeling (v6_UPP_model.py). Observations of CPU, Memory, Disk Space and Disk input/output are used as input features, while temperature is the target output.
