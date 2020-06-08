#!/usr/bin/env python

# ase.py
# Copyright (c) 2020. All rights reserved.

from typing import Sequence, TypeVar, Union, Dict
import os
import sys
import networkx
import numpy as np

from scipy.stats import norm
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import rbf_kernel

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import utils, container
from d3m.metadata import hyperparams, base as metadata_module, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult

from graspy.embed import AdjacencySpectralEmbed as graspyASE
from graspy.embed import OmnibusEmbed as graspyOMNI
from graspy.utils import pass_to_ranks as graspyPTR

Inputs = container.List
Outputs = container.List

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    max_dimension = hyperparams.Bounded[int](
        default=2,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = None
    )

    which_elbow = hyperparams.Bounded[int](
        default = 1,
        semantic_types= [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
        lower = 1,
        upper = 2
    )

    use_attributes = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = [
            'https://metadata.datadrivendiscovery.org/types/TuningParameter'
    ],
)

class AdjacencySpectralEmbedding(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Spectral-based trasformation of weighted or unweighted adjacency matrix.
    """
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_module.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': 'b940ccbd-9e9b-3166-af50-210bfd79251b',
        'version': "0.1.0",
        'name': "jhu.ase",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['ase primitive', 'graph', 'spectral', 'embedding', 'spectral method', 'adjacency', 'matrix'],
        'source': {
            'name': "JHU",
            'uris': [
                # Unstructured URIs. Link to file and link to repo in this case.
                'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/ase/ase.py',
                'https://github.com/neurodata/primitives-interfaces',
            ],
            'contact': 'mailto:hhelm2@jhu.edu'
        },
        'description': 'Spectral-based trasformation of weighted or unweighted adjacency matrix',
        'hyperparams_configuration': {
            'max_dimension': 'The maximum dimension that can be used for eigendecomposition',
            'which_elbow': 'The scree plot "elbow" to use for dimensionality reduction. High values leads to more dimensions selected.',
            'use_attributes': 'Boolean which indicates whether to use the attributes of the nodes.'
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
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # URIs at which one can obtain code for the primitive, if available.
        # 'location_uris': [
        #     'https://gitlab.com/datadrivendiscovery/tests-data/raw/{git_commit}/primitives/test_primitives/monomial.py'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #     ),
        # ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            "SINGULAR_VALUE_DECOMPOSITION"
        ],
        'primitive_family': "DATA_TRANSFORMATION",
        'preconditions': ['NO_MISSING_VALUES']
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0,
                 docker_containers: Dict[str, base.DockerContainer] = None) -> None:
        super().__init__(hyperparams=hyperparams,
                         random_seed=random_seed,
                         docker_containers=docker_containers)

    def produce(self, *, inputs: Inputs,
                timeout: float = None,
                iterations: int = None) -> CallResult[Outputs]:
        # unpacks necessary input arguments
        # note that other inputs are just passed through !
        learning_data = inputs[0]
        graphs_all = inputs[1]

        # ase only works for one graph (but we can change that)
        graph = graphs_all[0].copy()

        # catches link-prediction problem type
        # if it is not such - applies pass to ranks, which is a method to
        # rescale edge weights based on their relative ranks
        headers = learning_data.columns
        if "linkExists" in headers:
            graph_adjacency = np.array(graph.copy())
        else:
            graph_adjacency = graspyPTR(graph)

        n_elbows = self.hyperparams['which_elbow']
        max_dimension = self.hyperparams['max_dimension']

        n = graph_adjacency.shape[0]

        # it is counter-intuitive to embed into more dimensions than the original
        if max_dimension > n:
            max_dimension = n

        # check if there are any attributes, other than nodeIDs
        attributes_names = set([k for n in graph.nodes for k in graph.nodes[n].keys()])
        attributes_names.discard('nodeID')

        if self.hyperparams['use_attributes'] and len(attributes_names):

            # construct a matrix of attributes
            # TODO consider to just passing through the nodeList now that it exists
            attributes_array = np.zeros((n, len(attributes_names)))
            for i, attribute in enumerate(attributes_names):
                attributes_array[:, i] = np.array(list(networkx.get_node_attributes(graph, attribute).values()))

            # construct a graussian kernel. then apply pass to ranks.
            kernel_matrix = graspyPTR(rbf_kernel(attributes_array))

            adjacencies = [graph_adjacency, kernel_matrix]
            print(graph_adjacency, file=sys.stderr)
            print(kernel_matrix, file=sys.stderr)

            omni = graspyOMNI(n_components = max_dimension, n_elbows = n_elbows)
            embedding = np.array(omni.fit_transform(adjacencies))
            if embedding.ndim == 4: # directed inputs
                embedding = np.mean(embedding, axis=1)
                embedding = np.concatenate(embedding, axis=1)
            else: # undirected inputs
                embedding = np.mean(embedding, axis=0)

            print(embedding.shape)
            inputs[1][0] = container.ndarray(embedding)
            print("ase produce ended (omni used)", file=sys.stderr)

            return base.CallResult(inputs)

        else:
            ase_object = graspyASE(n_components=max_dimension,
                                   n_elbows = n_elbows)
            embedding = ase_object.fit_transform(graph_adjacency)
            if isinstance(embedding, tuple):
                embedding = np.concatenate(embedding, axis=1)

            inputs[1][0] = container.ndarray(embedding)

            print("ase produce ended (omni not used)", file=sys.stderr)

            return base.CallResult(inputs)
