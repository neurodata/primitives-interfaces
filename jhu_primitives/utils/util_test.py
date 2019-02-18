#! /usr/bin/env python

# util.py
# Created on 2017-09-14.

import os
import argparse
import importlib
import importlib.util
import sys
import json
import shutil
import re

"""
For reference, the pipelines that are functional are the following:
    gclass_ase_pipeline
    gclass_oosase_pipeline
    gclass_lse_pipeline
    gclass_ooslse_pipeline
    gmm_ase_pipeline
    gmm_oosase_pipeline
    gmm_lse_pipeline
    gmm_ooslse_pipeline
    sgc_pipeline
    sgm_pipeline
"""

PROBLEM_TYPES = [
    "graphMatching",
    "vertexNomination_class",
    "vertexNomination_clust",
    # "communityDetection"
    ]

DATASETS = {
            "graphMatching": [
                "49_facebook",
                # "LL1_Blogosphere_net",
                # "LL1_DIC28_net",
                # "LL1_ERDOS972_net",
                # "LL1_IzmenjavaBratSestra_net",
                # "LL1_REVIJE_net",
                # "LL1_SAMPSON_net",
                # "LL1_USAIR97_net",
                # "LL1_imports_net"
                ],
            "vertexNomination_class": [
                "LL1_net_nomination_seed",
                "LL1_EDGELIST_net_nomination_seed"
                ],
            "vertexNomination_clust": [
                "DS01876"
                ]# ,
            # "communityDetection": [
            #     "6_70_com_amazon",
            #     "6_86_com_DBLP",
            #     "LL1_Bio_dmela_net",
            #     "LL1_bn_fly_drosophila_medulla_net",
            #     "LL1_eco_florida_net"
            #     ]
            # "linkPrediction": [
            #     "59_umls"
            #     ]
            }

PIPELINES = {
            "graphMatching": [
                "sgm_pipeline",
                # "sgm_pipeline_10"
                ],
             "vertexNomination_class": [
                "gclass_ase_pipeline",
                "gclass_lse_pipeline",
                # "gclass_oosase_pipeline",
                # "gclass_ooslse_pipeline",
                "sgc_pipeline"
                ],
             "vertexNomination_clust": [
                "gmm_ase_pipeline",
                "gmm_lse_pipeline",
                # "gmm_oosase_pipeline",
                # "gmm_ooslse_pipeline",
                "sgc_pipeline"
                ]# ,
            # "communityDetection": [
            #     "gmm_oosase_pipeline",
            #     "gmm_ooslse_pipeline"
            #     ]
             }

DATASETS_THAT_MATCH_PROBLEM = [ "LL1_net_nomination_seed",
                                "49_facebook",
                                "DS01876",
                                "LL1_Blogosphere_net",
                                "LL1_DIC28_net",
                                "LL1_ERDOS972_net",
                                "LL1_IzmenjavaBratSestra_net",
                                "LL1_REVIJE_net",
                                "LL1_SAMPSON_net",
                                "LL1_USAIR97_net",
                                "LL1_imports_net",
                                "6_70_com_amazon",
                                "6_86_com_DBLP",
                                "LL1_Bio_dmela_net",
                                "LL1_bn_fly_drosophila_medulla_net",
                                "LL1_eco_florida_net"
                                ]


TRAIN_AND_TEST_SCHEMA_DATASETS = ["49_facebook",
                                "59_umls",
                                "DS01876",
                                "LL1_net_nomination_seed",
                                "6_70_com_amazon",
                                "6_86_com_DBLP"
                                ]


def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_args():
    parser = argparse.ArgumentParser(description = "Output a pipeline's JSON")
    parser.add_argument(
        'primitives_or_pipelines',
        action = 'store',
        help = "the type of object to generate jsons for",
    )
    arguments = parser.parse_args()
    return arguments.primitives_or_pipelines

def generate_json(type_):
    if type_ not in ['pipelines', 'primitives']:
        raise ValueError("Unsupported object type; 'pipelines' or 'primitives' only.")

    version = "v2019.1.21"
    # while version == "-1":
    #     version = input("Please select API version. \n0 for v2018.1.26 \n1 for v2018.4.18 \n2 for v2018.6.5\n3 for v2018.7.10 \n4 for v2019.1.21 \n")
    #     if version == "0":
    #         version = "v2018.1.26"
    #     elif version == "1":
    #         version = "v2018.4.18"
    #     elif version == "2":
    #         version = "v2018.6.5"
    #     elif version == "3":
    #         version = "v2018.7.10"
    #     elif version == "4":
    #         version = "v2019.1.21"
    path = os.path.join(os.path.abspath(os.getcwd()),"")

    if version == "v2019.1.21":
        jhu_path = os.path.join(path, "primitives-interfaces", "primitives_repo", version, "JHU")
    else:
        jhu_path = os.path.join(path, "primitives_repo", "archive", version, "JHU", "")

    all_primitives = os.listdir(jhu_path)
    primitive_names = [primitive.split('.')[-2] for primitive in all_primitives]

    versions = {}
    for primitive in all_primitives:
        versions[primitive] = os.listdir(os.path.join(jhu_path, primitive))[0]

    if type_ == 'primitives':
        for primitive in all_primitives:
            temp = jhu_path + primitive
            os.system('python -m d3m.index describe -i 4 ' + primitive + ' > ' + os.path.join(temp, versions[primitive], 'primitive.json'))
    else:
        python_paths = {}
        for primitive in all_primitives:
            temp_path = os.path.join(jhu_path, primitive, versions[primitive], 'pipelines', "")
            temp_dir = os.listdir(temp_path)
            python_paths[primitive.split('.')[-2]] = primitive

            for file in temp_dir:
                os.remove(temp_path + file)

        for problem_type in PROBLEM_TYPES:
            datasets = DATASETS[problem_type]
            pipelines = PIPELINES[problem_type]
            for pipeline in pipelines:
                path_to_pipeline = os.path.join(path, 'primitives-interfaces', 'jhu_primitives', 'pipelines', pipeline)

                spec = importlib.util.spec_from_file_location(pipeline, path_to_pipeline + '.py')

                module = importlib.util.module_from_spec(spec)

                spec.loader.exec_module(module)

                pipeline_class = getattr(module, pipeline)

                pipeline_dir = dir(module)
                p_dir = [convert(p) for p in pipeline_dir]
                primitives = [prim for prim in primitive_names if prim in p_dir]
                
                print(primitives)
                print(primitive_names)

                for dataset in datasets:

                    pipeline_object = pipeline_class()

                    # if dataset in DATASETS_THAT_MATCH_PROBLEM:
                    dataset_new = dataset + '_dataset'

                    with open('temp.json', 'w') as file:
                        text = pipeline_object.get_json()
                        file.write(str(text))

                    json_object = json.load(open('temp.json', 'r'))
                    pipeline_id = json_object['id']
                    primitive_id = json_object['steps'][-1]['primitive']['id']

                    for primitive in primitives:
                        temp_path = os.path.join(jhu_path, python_paths[primitive], versions[python_paths[primitive]], 'pipelines', "")
                        shutil.copy(os.path.join(path, 'temp.json'),
                                    os.path.join(jhu_path, python_paths[primitive], versions[python_paths[primitive]], 'pipelines', pipeline_id + '.json'))       # creates the pipeline json
                        write_meta(pipeline_id, dataset, dataset_new, temp_path + pipeline_id)
        os.remove('temp.json')

def write_meta(pipeline_id, dataset, dataset_new, path):
    meta = {}
    meta['problem'] = dataset + '_problem'
    meta['full_inputs'] = [dataset_new]
    if dataset in TRAIN_AND_TEST_SCHEMA_DATASETS:
        meta['train_inputs'] = [dataset_new + "_TRAIN"]
        meta['test_inputs'] = [dataset_new + "_TEST"]
        meta['score_inputs'] = [dataset_new + "_SCORE"]
    else:
        meta['train_inputs'] = [dataset_new]
        meta['test_inputs'] = [dataset_new]
        meta['score_inputs'] = [dataset_new]
    with open(path + '.meta', 'w') as file:
        json.dump(meta, file)

if __name__ == '__main__':
    type_ = load_args()
    generate_json(type_)


def file_path_conversion(abs_file_path, uri="file"):
    local_drive = abs_file_path.split(':')[0]
    try:
        file_path = abs_file_path.split(':')[1]
    except IndexError:
        file_path = local_drive
        local_drive = None
        return file_path

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
    elif local_drive is None:
        return s
    else:
        return local_drive + ":" + s

def data_file_uri(abs_file_path = "", uri = "file", datasetDoc = False, dataset_type = ""):
    if abs_file_path == "":
        raise ValueError("Need absolute file path ( os.path.abspath(os.getcwd()) )")
    local_drive, file_path = abs_file_path.split(':')[0], abs_file_path.split(':')[1]
    path_sep = file_path[0]
    file_path = file_path[1:]  # Remove initial separator
    if len(file_path) == 0:
        print("Invalid file path: len(file_path) == 0")
        return

    valid_type = False
    while not valid_type:
        type_ = input("Enter \n 0: exit \n 1: seed_datasets_current \n 2: training_datasets \n"
                          + " 3: if already in the data folder \n")
        if type_ == '0':
            return
        elif type_ == '1':
            data_dir = "datasets/seed_datasets_current"
            valid_type = True
        elif type_ == '2':
            data_dir = "datasets/training_datasets"
            valid_type = True
        elif type_ == '3':
            data_dir = ""
            valid_type = True
        else:
            print("Please enter 0, 1 or 2")

    valid_folder = False
    while not valid_folder:
        folder = input("Enter \n 0: exit \n Name of the data folder (case sensitive; must be in " + data_dir + ") \n")
        if folder == "0":
            return
        if type_ == '3':
            if os.path.isdir(folder):
                data_dir += folder
                valid_folder= True
        else:
            if os.path.isdir(data_dir  + "/" + folder):
                data_dir += "/" + folder
                valid_folder = True

    s = ""
    if path_sep == "/":
        splits = file_path.split("/")
        #data_folder = splits[-1]

    elif path_sep == "\\":
        splits = file_path.split("\\")
        #data_folder = splits[-1]
        for i in splits:
            if i != "":
                s += "/" + i
    else:
        print("Unsupported path separator!")
        return

    if datasetDoc:
        s = s + "/" + data_dir + "/"
        if dataset_type == "":
            s = s + folder + "_dataset/datasetDoc.json"
        elif dataset_type == "TRAIN":
            s = s + "/" + "TRAIN/dataset_TRAIN/datasetDoc.json"
        elif dataset_type == "TEST":
            s = s + "/" + "TEST/dataset_TEST/datasetDoc.json"
        else:
            raise ValueError('invalid dataset_type, use "" for the top level, "TRAIN" for the training dataset and "TEST" for thee test dataset')

    if uri == "file":
        return "file://localhost" + s
    else:
        return local_drive + ":" + s
