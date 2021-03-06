import os
import sys
import json
import typing
import numpy as np
import pandas as pd
import networkx as nx

from d3m import utils
from d3m import container
from d3m import exceptions
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult

import common_primitives

# from d3m.metadata import hyperparams, base as metadata_module, params

Inputs = container.Dataset
Outputs = container.List

class Hyperparams(hyperparams.Hyperparams):
    dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Resource ID of a DataFrame to extract if there are multiple tabular resources inside a Dataset and none is a dataset entry point.",
    )


class LoadGraphs(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which extracts a DataFrame out of a Dataset.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'cb192a83-63e2-4075-bab9-e6ba1a8365b6',
            'version': '0.1.0',
            'name': "Extract a list of Graphs from a Dataset",
            'python_path': 'd3m.primitives.data_transformation.load_graphs.JHU',
            'keywords': ['graph'],
            'source': {
                'name': "JHU",
                'uris': [
                    # Unstructured URIs. Link to file and link to repo in this case.
                    'https://github.com/neurodata/primitives-interfaces/blob/master/jhu_primitives/load_graphs/load_graphs.py',
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
        # obtain the path to dataset
        temp_json = inputs.to_json_structure()
        datasetDoc_uri = temp_json['location_uris'][0][7:]
        location_base_uri = '/'.join(datasetDoc_uri.split('/')[:-1])

        with open(datasetDoc_uri) as json_file:
            datasetDoc_json = json.load(json_file)
            dataResources = datasetDoc_json['dataResources']

        # get the task type from the task docs
        temp_path = datasetDoc_uri.split('/')
        problemDoc_uri = '/'.join(temp_path[:-2]) + '/' + '/'.join(temp_path[-2:]).replace('dataset', 'problem')

        with open(problemDoc_uri) as json_file:
             task_types = json.load(json_file)['about']['taskKeywords']

        # TODO consider avoiding explicit use of problem type throughout pipeline
        TASK = ""
        for task in task_types:
            if task in ["communityDetection", "linkPrediction", "vertexClassification", "graphMatching"]:
                TASK = task
        if TASK == "":
            raise exceptions.NotSupportedError("only graph tasks are supported")

        # load the graphs and convert to a networkx object
        graphs = []
        nodeIDs = []
        for i in dataResources:
            if i['resType'] == "table":
                if i['resID'] == 'learningData':
                    df = inputs['learningData']
                else:
                    node_list = pd.read_csv(location_base_uri + "/" + i['resPath'])

                    # assume it is a nodeList otherwise. currently, there
                    # aren't any D3M nodeList datasets that have more than one
                    # graph. furthermore, even if there was such, there isn't
                    # a way to match an edgeList to a nodeList. hence, we have
                    # to assume thatnodeList corresponds to the first graph
                    graph = graphs[0]

                    # the following block essentially catches VXTC synthetic
                    # dataset and overwrites nodeList indices withh edgeList.
                    # without a doubt not an AutoML way, but is necessary
                    first_idx_edge = str(sorted(list(graph.nodes(data=False)))[0])
                    first_idx_node = str(sorted(list(node_list['nodeID']))[0])
                    if (first_idx_edge.isdigit() and first_idx_node.isdigit()
                        and int(first_idx_edge) != int(first_idx_node)):
                        node_list = node_list.sort_values('nodeID').reset_index(drop=True)
                        d3m_indices = np.sort(np.array(list(graph.nodes(data=False))).astype(int))
                        node_list['nodeID'] = d3m_indices

                    # make nodeID an index (so it is not used an attribute)
                    node_list = node_list.set_index('nodeID')
                    node_list.index = node_list.index.astype(str)

                    # iterate over attributes and assign them to nodes
                    for attribute in node_list.columns.tolist():
                        series = pd.Series(node_list[attribute],
                                          index=node_list.index)
                        nx.set_node_attributes(graph,
                                               series.to_dict(),
                                               attribute)

            elif i['resType'] == 'graph':
                graph_temp = nx.read_gml(location_base_uri + "/" + i['resPath'])
                graphs.append(graph_temp)
                if TASK in ["communityDetection", "vertexClassification"]:
                    nodeIDs_temp = list(nx.get_node_attributes(graphs[0], 'nodeID').values())
                    nodeIDs_temp = np.array([str(i) for i in nodeIDs_temp])
                    nodeIDs_temp = container.ndarray(nodeIDs_temp)
                    nodeIDs.append(nodeIDs_temp)
            elif i['resType'] == "edgeList":
                temp_graph = self._read_edgelist(
                    location_base_uri + "/" + i['resPath'],
                    i["columns"], )
                graphs.append(temp_graph)
                if TASK in ["communityDetection", "vertexClassification"]:
                    nodeIDs_temp = list(temp_graph.nodes)
                    nodeIDs_temp = np.array([str(i) for i in nodeIDs_temp])
                    nodeIDs_temp = container.ndarray(nodeIDs_temp)
                    nodeIDs.append(nodeIDs_temp)

        return base.CallResult(container.List([df, graphs, nodeIDs, TASK]))


    def _read_edgelist(self, path, columns):
        # assumed that any edgelist passed has a source in the first col
        # and a reciever in the second col.
        # specify columns of edges
        from_column = columns[1]['colName']
        to_column = columns[2]['colName']
        # specify types
        dtypes_dict = {from_column: str, to_column: str}
        edgeList=pd.read_csv(path, dtype=dtypes_dict)
        G = nx.convert_matrix.from_pandas_edgelist(edgeList,
                                                   from_column,
                                                   to_column)
        return G


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
