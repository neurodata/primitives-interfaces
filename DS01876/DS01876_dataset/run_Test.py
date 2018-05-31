
import json
import numpy as np
import d3m
import jhu_primitives
from d3m.container import pandas as pd
from d3m.container import dataset as ds
import os
from d3m.metadata import hyperparams

from urllib import parse as url_parse

def file_path_conversion(abs_file_path, uri="file"):
    local_drive, file_path = abs_file_path.split(':')[0], abs_file_path.split(':')[1]
    path_sep = file_path[0]
    file_path = file_path[1:]  # Remove initial separator
    if len(file_path) == 0:
        print("Invalid file path: len(file_path) == 0")
        return

    s = ""
    if path_sep == "/":
        s = file_path
    elif path_sep == "\\":
        splits = file_path.split("\\")
        data_folder = splits[-1]
        for i in splits:
            if i != "":
                s += "/" + i
    else:
        print("Unsupported path separator!")
        return

    if uri == "file":
        return "file://localhost" + s
    else:
        return local_drive + ":" + s


absolute_url = "C://Users/joshu/Documents/Research/d3m_summer/DS01876/DS01876_dataset/datasetDoc.json"
file_path_conversion(abs_file_path=absolute_url)
datasetDoc_uri = file_path_conversion(abs_file_path=absolute_url)
parsed_uri = url_parse.urlparse(datasetDoc_uri)
parsed_uri
data = ds.D3MDatasetLoader().load(dataset_uri = datasetDoc_uri
                                  , dataset_id = 'DS01876_dataset'
                                  , dataset_version = '3.1.1'
                                  , dataset_name = "connectome graph data")

data = np.asmatrix(data)
hp = hyperparams.Hyperparameter
ASE = jhu_primitives.AdjacencySpectralEmbedding(hyperparams = hp )



ASE.produce(inputs=data)
