from networkx import Graph
import networkx as nx
import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import os

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    embeddings: container.List
    inner_products: container.List

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None

class LinkPredictionRankClassifier(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '25e97696-b96f-4f5c-8620-b340fe83414d',
        'version': "0.1.0",
        'name': "jhu.link_pred_rc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.LinkPredictionRankClassifier',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'inner product'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/link_pred_rc/link_pred_rc.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        'installation': [
            {
            'type': 'UBUNTU',
            'package': 'libxml2-dev',
            'version': '2.9.4'
            },
            {
            'type': 'UBUNTU',
            'package': 'libpcre3-dev',
            'version': '2.9.4'
            },
            {
            'type': 'PIP',
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),),
            },
            ],
        'algorithm_types': [
            "SINGULAR_VALUE_DECOMPOSITION" # need to find an appropriate tag
        ],
        'primitive_family': "DATA_TRANSFORMATION",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)
        self._fitted: bool = False
        self._inner_products: container.List = []
        self._embeddings: container.List = []

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        if not self._fitted:
            raise ValueError("Not fitted")
        
        print(inputs)

        return base.CallResult(inputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        K = len(self._training_inputs) # K = number of link types

        self._embeddings = [self._training_inputs[i][0] for i in range(K)]
        self._embeddings = container.List(self._embeddings)

        n_info_each_graph = np.zeros(len(self._training_inputs))
        n_edges_each_graph = np.zeros(len(self._training_inputs))

        for i in range(K):
            for j in range(len(self._training_inputs[i][1])):
                n_info_each_graph[i] += 1
                n_edges_each_graph[i] += int(self._training_inputs[i][1][j][2])

        inner_products = [[np.zeros(int(n_info_each_graph[i] - n_edges_each_graph[i])), np.zeros(int(n_edges_each_graph[i]))] for i in range(K)]

        for i in range(K):
            zeros, ones = 0, 0
            for j in range(len(self._training_inputs[i][1])):
                temp_node_1 = int(self._training_inputs[i][1][j][0])
                temp_node_2 = int(self._training_inputs[i][1][j][1])
                temp_class = int(self._training_inputs[i][1][j][2])

                if temp_class == 0:
                    inner_products[i][temp_class][zeros] = self._embeddings[i][temp_node_1] @ self._embeddings[i][temp_node_2]
                    zeros += 1
                else:
                    inner_products[i][temp_class][ones] = self._embeddings[i][temp_node_1] @ self._embeddings[i][temp_node_2]
                    ones += 1

        self._inner_products = container.List()

        for i in range(K):
            sorted_class0 = np.sort(inner_products[i][0])
            sortec_class1 = np.sort(inner_products[i][1])
            self._training_inputs[i][1] = container.List([container.ndarray(inner_products[i][0]), container.ndarray(inner_products[i][1])]) # replacing training data info with lists
            self._inner_products.append(self._training_inputs[i][1])

        self._fitted = True

        return base.CallResult(None)

    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")

        return Params(
            inner_products = self._inner_products,
            embeddings = self._embeddings
        )

    def set_params(self, *, params: Params) -> None:
        self._fitted = True
        self._inner_products = params['inner_products']
        self._embeddings = params['embeddings']
