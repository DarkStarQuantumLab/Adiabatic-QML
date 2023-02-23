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
from dimod import Binary, BQM
import dimod, re
from neal import SimulatedAnnealingSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import DWaveSampler
from dimod.binary import quicksum
from typing import Tuple, Any
import itertools

class QAOptimizer:
    """Quantum Annealing optimizer.
        
       Minimizes a loss function as optimization problem, QUBO, on a quantum annealer. 
    """

    def __init__(self, dimention, p_vec_length, backend):
        self.backend = backend
        self.dimention = dimention
        self.p_vec_length = p_vec_length

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

        num_itereations = self.dimention*self.p_vec_length    
        for i in range(num_itereations):
            quadratic_terms.append([quantum_vars[i] * quadratic[i][j] * quantum_vars[j] for j in range(num_itereations)])

        quadratic_terms = list(itertools.chain(*quadratic_terms))

        for i in range(num_itereations):
            linear_terms.append(quantum_vars[i]*linear[i])

        quantum_model = dimod.CQM()

        quadratic_objective = quicksum(quadratic_terms)
        linear_objective = quicksum(linear_terms)

        quantum_model.set_objective(quadratic_objective + linear_objective)

        bqm, _ = dimod.cqm_to_bqm(quantum_model, lagrange_multiplier=1)

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
        quantum_vars = {(i):Binary(f'w{i}') for i in range(self.p_vec_length * self.dimention)}
        # construct binary model
        objective_to_minimize = self._get_objective(quadratic, linear, quantum_vars)

        if self.backend =='SA': # optimize as Qauntum Annealing
            sampler_simulated_annealing = SimulatedAnnealingSampler()
            sampleset_neal = sampler_simulated_annealing.sample(objective_to_minimize, num_reads=1000)
            samples = sampleset_neal.aggregate()
        elif self.backend == 'QPU': #optimize on a quantum hardware
            sampler = DWaveSampler()
            sampler = EmbeddingComposite(sampler)
            samples = sampler.sample_qubo(objective_to_minimize, chain_strength=15, num_reads=1000)
        else:
            raise ValueError("""{self.backend} is unsupported type of beckends. Use SA for Simulated Annealing 
                            or QPU to submit to a quantum hardware.""")
        pass

        return samples
        