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

import itertools
import numpy as np
from linear.annealing_optimizer import QAOptimizer


class QALinearRegression():
    """ Implementation of the Least Squares Linear Regression on Qauntum Annealers.

        y_i = x_0 + w_i * x_i.

        The QALinearRegression fits the linear model with coeficients w_n to minimize the 
        sum of squares between the observed data in the dataset and predicted data.
        The minimization is submited to Simulated Annealing (classical part) or QPU.
    """

    def __init__(self, normalize=False):
        self.normalize = normalize
        self.type = 'linear'
        self.is_trained = False
    
    def _check_is_trained(self):
        """Validate a model training before making a prediction."""

        if not self.is_trained:
            raise Exception("Train the model before prediction. \
                            model = QALinearRegression \
                            model.train(x_points, y_labels)")

    def train(self, x:np.array, y:np.array, backend='SA'):
        """ Train the QA linear model.

            Parameters:
                x: array-like training data.
                y: array-like labels data.
                backend: backend to minimize loss function. 
                    SA (default) minimizes the least squares as Simulated Annealing.
                    QPU submits to a quantum hardware (access to a quantum hardware is required).  
            
            Returns:
                None.
        """
        if len(x) != len(y):
            raise ValueError("Input variables with inconsistent number of samples")
        if x.ndim != 2:
            raise ValueError("""Expected 2D array, got 1D array instead. 
                            Reshape your data either using array.reshape(-1, 1) 
                            if your data has a single feature, or array.reshape(1, -1) 
                            if it contains a single sample.""")
        _, self.num_features = x.shape

        #TODO: handle normalization
        if self.normalize:
            pass

        dimention = x.ndim
        array_length = len(x)

        # define precision vector and precision matrix
        # for artificial data
        p_vector = np.array([0.25, 0.25, 0.5, 0.5, 0.75])

        # issue with dimention
        # temporarly, set dimention to 2, change later
        dimention = 2 
        identity = np.identity(dimention)
        precision_matrix = np.kron(identity, p_vector)

        # prepare data to train the model on a quantum annealer
        augment_ones = np.ones(array_length)
      
        x = np.reshape(x, -1)
        x_quantum = np.vstack((augment_ones, x))
        
        # create regression coefficients as matrix multiplication
        # here, regression vector is the QUBO matrix submitted to DWave's BQM model 
        regression_vector = np.transpose(precision_matrix) @ x_quantum @ np.transpose(x_quantum) @ precision_matrix
        quadratic = np.triu(2*regression_vector) #quadratic coeff. 

        # linear part of the QUBO model
        linear_vector = np.transpose(precision_matrix) @ x_quantum @ np.transpose(y)
        linear = -2.0 * linear_vector

        optimizer = QAOptimizer(len(y), backend=backend)
        binary_set_of_weights = optimizer.minimize_loss(quadratic, linear)

        # currently, the most optimized set of weights selected, 
        # think how to emplement as the range of linear models. 

        minimized_weights = precision_matrix @ binary_set_of_weights.record[0][0]

        self.model_weights = minimized_weights
        self.is_trained = True


    def predict(self, x:np.array) -> np.array:
        """
        Make prediction on data.
        Params:
            x: the array of data to make prediction on. 
        Returns:
            y_predicted: an array of predicted values.
        """

        self._check_is_trained()

        _, num_rows = x.shape
        if num_rows != self.num_features:
            raise ValueError("X has {} but QALinearRegression expects {} as input.".format(num_rows, self.num_features))

        # print("self.model_weights = ", self.model_weights)
        weight = np.array(self.model_weights[1])
        y_predicted = weight * x
       
        return y_predicted

    def r_score(self, y_act:np.array, y_pred:np.array) -> float:
        """
        Calculates the Coefficient of Determination using numpy.
        Args:
            y_act: (np.array) actual data points
            y_pred: (np.array) predicted by model datapoints
        Return:
            r_score: Square root of the Coefficient of Determination.
        """

        corr_matrix = np.corrcoef(y_act, y_pred.reshape(1, len(y_act)))
        corr = corr_matrix[0,1]
        return corr**2

    def mse(self, y_act:np.array, y_pred:np.array) -> float:
        """
        Calculates the Mean Square Error (MSE) using numpy.
        Args:
            y_act: (np.array) actual data points
            y_pred: (np.array) predicted by model datapoints
        Return:
            mse: calculated MSE score.
        """
        mse = (np.linalg.norm(y_act - y_pred)**2)/len(y_act)
        return mse

    def root_mse(self, y_act:np.array, y_pred:np.array) -> float:
        """
        Calculates the Root Mean Square Error (RMSE) using numpy.
        Args:
            y_act: (np.array) actual data points
            y_pred: (np.array) predicted by model datapoints
        Return:
            root_mse: calculated squre root of MSE
        """
        mse = (np.linalg.norm(y_act - y_pred)**2)/len(y_act)
        return np.sqrt(mse)

    def mae(self, y_act:np.array, y_pred:np.array) -> float:   
        """
        Calculates the Mean Absolute Error (MAE) using numpy.
        Args:
            y_act: (np.array) actual data points
            y_pred: (np.array) predicted by model datapoints
        Return:
            mae: calculated MAE score
        """
        mae = (np.linalg.norm(np.abs(y_act - y_pred))) / len(y_act)
        return mae

