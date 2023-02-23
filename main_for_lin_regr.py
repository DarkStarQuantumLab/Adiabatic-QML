# Copyright DarkStarQuantumLab, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import sklearn.datasets as dt
from qml_adiabatic import QALinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os
import imageio

if __name__ == "__main__":
    
    # define a random dataset for a qa linear regression to test
    seed = 20
    rand_state = 11
    rand= np.random.RandomState(seed)   
    x, y = dt.make_regression(n_samples=100, n_features=1, noise=5, n_informative=1, 
                                            random_state=rand_state)

    x_points, x_test, y_labels, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    # define a precision vector
    precision_vector = np.array([0, 0.125, 0.25, 0.5, 0.75])    

    # train adiabatic linear regression
    qa_lin_model = QALinearRegression(precision_vector=precision_vector, normalize=True, scaler="MaxAbsScaler")
    qa_lin_model.train(x_points, y_labels)
    y_predicted = qa_lin_model.predict(x_points)
    # print("Predicted values: {}".format( y_predicted))
    print("Adiabatic Linear Regression r^2 score on training data: {}".format(qa_lin_model.r_score(y_labels, y_predicted)))

    # train the sklearn linear regression
    model = LinearRegression()
    trained_model = model.fit(x_points, y_labels)
    y_predict_classical = trained_model.predict(x_points)
    print("Sklearn Linear Regression r^2 score on training data: {}".format(r2_score(y_labels, y_predict_classical)))

    # plot and save file
    plt.scatter(x_points, y_labels, color="purple")
    plt.plot(x_points, y_predict_classical, color='pink', linewidth=3, label='Classical')
    plt.plot(x_points, y_predicted, color='green', linewidth=2, label='Quantum')
    plt.legend(loc='upper left')
    plt.title("Adiabatic vs Classical Linear Regression on Artificial Data")
    plt.savefig("results/artificial.png")
    plt.show()

    # model verification.
    print("Evaluating Linear Regressions . . .")
    y_predict_classical = trained_model.predict(x_test)

    y_predicted_quantum = qa_lin_model.predict(x_test)

    # print metrics
    print('Adiabatic Regression MAE on test data = ', qa_lin_model.mae(y_test, y_predicted_quantum))
    print('Adiabatic Regression MSE on test data = ', qa_lin_model.mse(y_test, y_predicted_quantum))
    print('Adiabatic Regression Root MSE on test data = ', qa_lin_model.root_mse(y_test, y_predicted_quantum))
    print('Adiabatic Regression R^2 on test data= ', qa_lin_model.r_score(y_test, y_predicted_quantum))
    print("Sklearn R^2 on test data = ", qa_lin_model.r_score(y_test, y_predict_classical))
