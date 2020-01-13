from networkx import Graph
import networkx as nx
import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import os
import sys

from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

Inputs = container.List
Outputs = container.DataFrame

class Params(params.Params):
    embeddings: container.List
    inner_products: container.List

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None

class LinkPredictionRankClassifier(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A primitive that predicts the existence of a link if it falls within the interquartile range of
    inner products.
    """

    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '25e97696-b96f-4f5c-8620-b340fe83414d',
        'version': "0.1.0",
        'name': "jhu.link_pred_rc",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.link_prediction.rank_classification.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'inner product'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/link_pred_rc/link_pred_rc.py',
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
            "HEURISTIC"
        ],
        'primitive_family': "LINK_PREDICTION",
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
            
        np.random.seed(self.random_seed)
        
        csv = inputs[1]
        

        # print(csv, file=sys.stderr)
        csv_headers = csv.columns
        for header in csv_headers:
            if header[:6] == "source":
                SOURCE = header
            elif header[:6] == "target":
                TARGET = header
        
        source_nodeID = np.array(csv[SOURCE]).astype(int)
        target_nodeID = np.array(csv[TARGET]).astype(int)
        
        try:
            int(np.array(csv['linkType'])[0])
        except:
            csv['linkType'] = np.zeros(len(source_nodeID))
        
        link_types = np.array(csv['linkType']).astype(int)

        n_links = len(self._inner_products) - 1
        n_nodes = int(self._embeddings.shape[0] / n_links)

        n_preds = csv.shape[0]

        predictions = np.zeros(n_preds)

        global_noexists = self._inner_products[-1][0]
        global_exists = self._inner_products[-1][1]

        # The following code is used for "global" classification only; i.e. we ignore edge type training data
        for i in range(n_preds):
            temp_source = source_nodeID[i]
            temp_target = target_nodeID[i]
            temp_link = link_types[i]
            temp_inner_product = self._embeddings[temp_link*n_nodes + temp_source-1] @ self._embeddings[temp_link*n_nodes + temp_target-1]
            temp_noexists = self._inner_products[temp_link][0]
            temp_exists = self._inner_products[temp_link][1]

            # There are three 'degenerate' cases --
            # 1) Both the exists and no exists lists are empty (first 'if')
            # 2/3) One but not the other is empty ('elif')
            # if len(temp_noexists) == 0 and len(temp_exists) == 0:
            rank_noexists = np.sum(temp_inner_product > global_noexists)
            quantile_noexists = rank_noexists / len(global_noexists)

            rank_exists = np.sum(temp_inner_product > global_noexists)
            quantile_exists = rank_exists / len(global_exists)                  

            if abs(quantile_noexists - 1/2) < abs(quantile_exists - 1/2):
                predictions[i] = int(0)
            elif abs(quantile_noexists - 1/2) > abs(quantile_exists - 1/2):
                predictions[i] = int(1)
            else:
                predictions[i] = int(np.random.binomial(1, 0.5))
            
        csv['linkExists'] = predictions.astype(int)
        outputs = container.DataFrame(csv[['d3mIndex', 'linkExists']])

        return base.CallResult(outputs)

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        if self._fitted:
            return base.CallResult(None)

        embeddings = self._training_inputs[0]
        csv = self._training_inputs[1]
        n_nodes, n_links = self._training_inputs[2][0], self._training_inputs[2][1]

        n_info = csv.shape[0]
        ranks = [[[], []] for i in range(n_links + 1)]

        try:
            int(np.array(csv['linkType'])[0])
        except:
            csv['linkType'] = np.zeros(n_info)

        # print(csv, file=sys.stderr)
        csv_headers = csv.columns
        for header in csv_headers:
            if header[:6] == "source":
                SOURCE = header
            elif header[:6] == "target":
                TARGET = header

        for i in range(n_info):
            temp_link = int(np.array(csv['linkType'])[i])
            temp_exists = int(np.array(csv['linkExists'])[i])
            temp_source = int(np.array(csv[SOURCE])[i])
            temp_target = int(np.array(csv[TARGET])[i])
            temp_dot = embeddings[temp_link*n_nodes + temp_source - 1] @ embeddings[temp_link*n_nodes + temp_target - 1]
            ranks[temp_link][temp_exists].append(temp_dot)
            ranks[-1][temp_exists].append(temp_dot)

        for i in range(len(ranks)):
            ranks[i][0] = np.sort(ranks[i][0])
            ranks[i][1] = np.sort(ranks[i][1])

        self._embeddings = embeddings
        self._inner_products = ranks

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
