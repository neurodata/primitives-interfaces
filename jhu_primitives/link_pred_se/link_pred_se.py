from networkx import Graph
import networkx as nx
import numpy as np
from typing import Sequence, TypeVar, Union, Dict
import os

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container
from d3m import utils
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from ..ase import AdjacencySpectralEmbedding
from ..ase.ase import Hyperparams as ASE_Hyperparams

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    #dim = hyperparams.Hyperparameter[None](default=None)
    dim = None
    # ASE Hyperparams
    # List of [adjacency, laplacian]

class LinkPredictionSpectralEmbedder(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': '09f2eea8-667c-44b8-a955-6a153ba9ccc3',
        'version': "0.1.0",
        'name': "jhu.link_pred_se",
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.jhu_primitives.LinkPredictionSpectralEmbedder',
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['graph', 'spectral embedding'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/jhu_primitives/link_se/link_se.py',
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
            "SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        
        ase_hp = ASE_Hyperparams.defaults()

        for i in range(len(inputs)):
            inputs[i] = AdjacencySpectralEmbedding(hyperparams=ase_hp).produce(inputs = inputs[i]).value

        return base.CallResult(inputs)
