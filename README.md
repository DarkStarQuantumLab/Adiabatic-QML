# Overview

The Adiabatic QML repository contains the implementation of a Linear Regression on [DWave Quantum](https://github.com/dwavesystems) Annealer as described in Date, P. and Potok, T. "Adiabatic quantum linear regression". [image](results/artificial_data.png)


The code implementation is similar to the classical ML frameworks. However, the loss-function optimization has been adopted and implemented in such a way that it could be submitted to DWave Quantum Annealers. The Qauntum Annealer minimizes the loss and returns a distribution of possible linear regressions. The algorithm has been tested on the Simulated Annealing only.

# Usage

The Addiabatic Linear Regression model is used in a similar way as its counterparts avaliable in Scikit-learn framework. The code implementation is avaliable in <annealing_lin_regression.py>, <annealing_optimizer.py> files in <linear> subdirectory.

The following line initializes the model as Addiabatic Linear Regression:

    addiabatic_lin_model = QALinearRegression()

To train the model, run:

    addiabatic_lin_model.train(x, y)

Finally, to make prediction:

    predicted_data = addiabatic_lin_model.predict(x)

*Note:* Given the probabilistic nature of Qauntum Annealers, we have to controll an exta parameter defined as "precision vector" and described in the research paper by P. Date and T. Protok. 

All steps are compossed in <main_for_lin_regr.py> file. To obtain the results of the Quantum Linear Regression, simple execute the code in <main_for_lin_regr.py> file:

    python3  main_for_lin_regr.py

# Results 

Adiabatic Linear Regression Evaluation Metrics on artificial data.
MAE of the model is 135.74.
MSE of the model is 92121.85.
RMSE of the model is 303.52.
R^2 of the model is 0.999992.

# License
This work is licensed under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) and is owned by [DarkStarQuantumLab](https://github.com/DarkStarQuantumLab). 

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# References.
1. Date, P., Potok, T. Adiabatic quantum linear regression. Sci Rep 11, 21905 (2021). https://doi.org/10.1038/s41598-021-01445-6