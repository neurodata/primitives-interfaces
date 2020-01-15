import os
import typing
import sys

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import common_primitives

# from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
# from d3m import container
from d3m import utils
# from d3m.metadata import hyperparams, base as metadata_module, params
# from d3m.primitive_interfaces import base
# from d3m.primitive_interfaces.base import CallResult

# Inputs = container.Dataset
# Outputs = container.List

Inputs = container.Dataset
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.",
    )


class DatasetToGraphList(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts a DataFrame out of a Dataset.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'cb192a83-63e2-4075-bab9-e6ba1a8365b6',
            'version': '0.1.0',
            'name': "Extract a list of Graphs from a Dataset",
            'python_path': 'd3m.primitives.data_transformation.dataset_to_graph_list.JHU',
            'keywords': ['graph'],
            'source': {
                'name': "JHU",
                'uris': [
                    # Unstructured URIs. Link to file and link to repo in this case.
                    'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/dataset_to_graph_list/dataset_to_graph_list.py',
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
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
        print('dataset to graph list, baby!!', file=sys.stderr)
        print(dir(inputs), file=sys.stderr)
        print(inputs.keys(), file=sys.stderr)
        print(inputs.to_json_structure(), file=sys.stderr)

        data_resources_keys = list(inputs.keys())

        for resource_id in data_resources_keys:
            if resource_id == 'learningData':
                learningData=inputs[resource_id]
            elif:
                pass


        # dataframe_resource_id, dataframe = base_utils.get_tabular_resource(inputs, self.hyperparams['dataframe_resource'])

        # dataframe.metadata = self._update_metadata(inputs.metadata, dataframe_resource_id)

        # assert isinstance(dataframe, container.DataFrame), type(dataframe)

        return base.CallResult(dataframe)


    # TODO: not sure what this does or if its relevant to graph problems.
    def _update_metadata(self, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update(
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            },
        )

        new_metadata = metadata_base.DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata