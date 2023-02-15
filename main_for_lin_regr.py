# author: Olga Okrut. Email: vokrut42sv@gmail.com

import numpy as np
import sklearn.datasets as dt
from qml_adiabatic import QALinearRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # small test on artificial data. Change the precision vector before running.

    # define a random dataset for a qa linear regression to test
    seed = 11
    rand_state = 11
    rand= np.random.RandomState(seed)   
    x_points, y_labels = dt.make_regression(n_samples=10, n_features=1, noise=1, random_state=rand_state)

    # define a qa linear regression
    qa_lin_model = QALinearRegression()
    qa_lin_model.train(x_points, y_labels)
    y_predicted = qa_lin_model.predict(x_points)
    print("y_predicted = ", y_predicted)

    model = LinearRegression()
    trained_model = model.fit(x_points, y_labels)
    y_predict_classical = trained_model.predict(x_points)
    # # plot and save file
    plt.scatter(x_points, y_labels)
    plt.plot(x_points, y_predict_classical, color='pink', linewidth=3, label='Classical')
    plt.plot(x_points, y_predicted, color='green', linewidth=1, label='Quantum')
    plt.legend(loc='upper left')
    plt.title("Adiabatic vs Classical Linear Regression on Artificial Data")
    plt.savefig("results/artificial.png")
    plt.show()

    # model verification.
    print("Evaluate Quantum Linear Regression . . .")
    x_test, y_actual = dt.make_regression(n_samples=5, n_features=1, noise=1, random_state=rand_state)

    y_predicted = qa_lin_model.predict(x_test)

    # print metrics
    print('MAE = ', qa_lin_model.mae(y_actual, y_predicted))
    print('MSE = ', qa_lin_model.mse(y_actual, y_predicted))
    print('Root MSE = ', qa_lin_model.root_mse(y_actual, y_predicted))
    print('R^2 = ', qa_lin_model.r_score(y_actual, y_predicted))