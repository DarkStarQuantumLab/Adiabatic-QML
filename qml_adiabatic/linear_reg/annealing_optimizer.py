# author: Olga Okrut. Email: vokrut42sv@gmail.com

import numpy as np
from dimod import Binary, BQM
import dimod, re
from neal import SimulatedAnnealingSampler
from dimod.binary import quicksum
from typing import Tuple, Any
import itertools


class QAOptimizer:
    """Quantum Annealing optimizer 
    """

    def __init__(self, num_of_vars, backend):
        self.backend = backend
        self.num_of_vars = num_of_vars

    def _get_objective(self, quadratic:np.array, linear:np.array, quantum_vars:dimod.BinaryQuadraticModel) -> dimod.BQM:
        """Constructs objective function to be optimized.
            Params:
                quantum_vars: Binary quantum variables representing weights of the regression model
            Returns:
                objective function to be minimized.
        """
        quadratic_terms = []
        linear_terms = [] 

        keys = quantum_vars.keys()
        values = (quantum_vars[key] for key in keys)
        comb = itertools.combinations(zip(keys,values), 2)
      
        for pair in comb:
            i = pair[0][0]
            j = pair[1][0]
            term = pair[0][1]*pair[1][1]*quadratic[i][j]
            quadratic_terms.append( pair[0][1]*pair[1][1]*quadratic[i][j])

        for i in range(self.num_of_vars):
            linear_terms.append(quantum_vars[i]*linear[i])

        quantum_model = dimod.CQM()

        quadratic_objective = quicksum(quadratic_terms)
        linear_objective = quicksum(linear_terms)

        quantum_model.set_objective(quadratic_objective + linear_objective)

        bqm, invert = dimod.cqm_to_bqm(quantum_model, lagrange_multiplier=1)

        return bqm

    def minimize_loss(self, quadratic:np.array, linear:np.array) -> dimod.SampleSet:
        """
            Minimizes the loss function (least squares).

            Parameters:
                quadratic: quadratic coefficients of the QUBO model.
                linear: linear coefficients of the QUBO model.

            Returns: 
                A ``dimod.SampleSet``, a set of binary solution representing best choise of binaries for weights.
        """

        # create binary variables for models weights
        quantum_vars = {(i):Binary(f'w{i}') for i  in range(self.num_of_vars) }
        # construct binary model
        objective_to_minimize = self._get_objective(quadratic, linear, quantum_vars)

        if self.backend =='SA': # optimize as Qauntum Annealing
            sampler_simulated_annealing = SimulatedAnnealingSampler()
            sampleset_neal = sampler_simulated_annealing.sample(objective_to_minimize, num_reads=1000)
            samples = sampleset_neal.aggregate()
        elif self.backend == 'QPU': #optimize on a quantum hardware
            pass
        else:
            raise ValueError("""{self.backend} is unsupported type of beckends. Use SA for Simulated Annealing 
                            or QPU to submit to a quantum hardware.""")
        pass

        return samples
        