from typing import Sequence, TypeVar, Union, Dict
import os
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
import numpy as np

from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from graspy.cluster.gclust import GaussianCluster as graspyGCLUST

Inputs = container.List
Outputs = container.DataFrame

class Params(params.Params):
    embedding : container.ndarray

class Hyperparams(hyperparams.Hyperparams):
    max_clusters = hyperparams.Bounded[int](
        default = 2,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
        ],
        lower = 2,
        upper = None
    )

class SingleGraphVertexNomination(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params,Hyperparams]):
    """
    Expecation-Maxmization algorithm for clustering
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'dad3e96a-88a3-4e96-ba38-c152c210912f',
        'version': "0.1.0",
        'name': "jhu.1gvn",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.vertex_nomination.spectral_vertex_nomination.JHU',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'gaussian clustering', 'vertex nomination'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/1gvn/1gvn.py',
#                'https://github.com/youngser/primitives-interfaces/blob/jp-devM1/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces.git',
            ],
            'contact': 'mailto:hhelm2@jhu.edu',
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
            'type': metadata_module.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/neurodata/primitives-interfaces.git@{git_commit}#egg=jhu_primitives'.format(
            git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        'description': 'ingle graph vertex nomination via hierarchical clustering and Expecation-Maxmization.',
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris':
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "EXPECTATION_MAXIMIZATION_ALGORITHM"
        ],
        'primitive_family': "VERTEX_NOMINATION",
        'preconditions': ['NO_MISSING_VALUES']
        })
    
    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

        self._embedding: container.ndarray = None
            
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        if self._embedding is None:
            self._embedding = inputs[0]
            
        N, d = self._embedding.shape

        nodeIDs = inputs[1]
        nodeIDS = np.array([int(i) for i in nodeIDs])

        max_clusters = self.hyperparams['max_clusters']

        if max_clusters < self._embedding.shape[1]:
            self._embedding = self._embedding[:, :max_clusters].copy()

        gclust_object = graspyGCLUST(max_components=max_clusters, covariance_type="all")
        gclust_object.fit(self._embedding)
        model = gclust_object.model_
        
        pis, means, precs = model.weights_, model.means_, model.precisions_

        predictions = model.predict(self._embedding)
        
        D = np.zeros(shape=(N, N))
        
        if d == 1:
            cluster_label = predictions[i]
            i_embedding = self._embedding[i]
            for j in range(i + 1, N):
                j_embedding = self._embedding[j]
                eucl_dist = i_embedding - j_embedding
                Mahal_dist = eucl_dist * precs[cluster_label] * eucl_dist
                D[i, j] = Mahal_dist
                D[j, i] = Mahal_dist
        else:
            for i in range(N):
                cluster_label = predictions[i]
                i_embedding = self._embedding[i]
                for j in range(i + 1, N):
                    j_embedding = self._embedding[j]
                    eucl_dist = i_embedding - j_embedding
                    Mahal_dist = eucl_dist @ precs[cluster_label] @ eucl_dist[None].T
                    D[i, j] = Mahal_dist[0]
                    D[j, i] = D[i, j]
                
        D_idx = np.zeros(shape=(N, N-1))
        for i in range(N):
            D_idx[i] = np.argsort(D[i])[1:]

        columns = ['match%i'%(i + 1) for i in range(N - 1)]
    
        # May need to create nodeID <-> d3m index map 
        output = container.DataFrame(D_idx, index = nodeIDs, columns = columns).astype(int)
        output.index.name = "d3mIndex"

        return base.CallResult(output)
    def set_training_data(self, *, inputs: Inputs) -> None:
        self._training_inputs = inputs

    def get_params(self) -> Params:
        return Params(embedding = self._embedding)

    def set_params(self, *, params: Params) -> None:
        self._embedding = params['embedding']

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        return base.CallResult(None)

