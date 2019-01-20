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

"""
For reference, the pipelines that are functional are the following:
    gclass_ase_pipeline
    gclass_lse_pipeline
    gmm_ase_pipeline
    gmm_lse_pipeline
    sgc_pipeline
    sgm_pipeline
"""
PROBLEM_TYPES = ['graphMatching', 'vertexNomination_class', 'vertexNomination_clust']

DATASETS = {'graphMatching': ['49_facebook_problem_TRAIN',
                              'LL1_Blogosphere_net_problem',
                              'LL1_DIC28_net_problem',
                              'LL1_ERDOS972_net_problem',
                              'LL1_IzmenjavaBratSestra_net_problem',
                              'LL1_REVIJE_net_problem',
                              'LL1_SAMPSON_net_problem',
                              'LL1_USAIR97_net_problem',
                              'LL1_imports_net_problem'],
            'vertexNomination_class': ['LL1_EDGELIST_net_nomination_seed_problem_TRAIN',
                                       'LL1_net_nomination_seed_problem_TRAIN'],
            'vertexNomination_clust': ['DS01876_problem_TRAIN'],
            }
#'communityDetection': ['6_70_com_amazon_problem_TRAIN',
#                       '6_86_com_DBLP_problem_TRAIN',
#                       'LL1_Bio_dmela_net_problem',
#                       'LL1_bn_fly_drosophila_medulla_net_problem',
#                       'LL1_eco_florida_net_problem],
#'linkPrediction': ['59_umls_problem_TRAIN']


PIPELINES = {'graphMatching': ['sgm_pipeline'],
             'vertexNomination_class': ['gclass_ase_pipeline',
                                        'gclass_lse_pipeline',
                                        'sgc_pipeline'],
             'vertexNomination_clust': ['gmm_ase_pipeline',
                                        'gmm_lse_pipeline',
                                        'sgc_pipeline'],
             }

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
    
    version = "-1"

    while version == "-1":
        version = input("Please select API version. \n0 for v2018.1.26 \n1 for v2018.4.18 \n2 for v2018.6.5\n3 for v2018.7.10 \n")
        if version == "0":
            version = "v2018.1.26"
        elif version == "1":
            version = "v2018.4.18"
        elif version == "2":
            version = "v2018.6.5"
        elif version == "3":
            version = "v2018.7.10"

    path = os.path.abspath(os.getcwd()) + "\\"
    jhu_path = path + "primitives_repo\\" + version + "\\JHU\\"

    all_primitives = os.listdir(jhu_path)
    d3m_string = ""

    for i in all_primitives[0].split('.')[:-1]:
        d3m_string = d3m_string + i + "."

    primitive_names = [primitive.split('.')[-1] for primitive in all_primitives]

    versions = {}

    for i in range(len(all_primitives)):
        versions[primitive_names[i]] = os.listdir(jhu_path + all_primitives[i])[0]

    if type_ == 'primitives':
        for i in range(len(all_primitives)): 
            temp = jhu_path + all_primitives[i]
            os.system('python -m d3m.index describe -i 4 ' + all_primitives[i] + ' > ' + temp + '\\' + versions[primitive_names[i]] + '\\primitive.json')
    else:
        for problem_type in PROBLEM_TYPES:
            datasets = DATASETS[problem_type]
            pipelines = PIPELINES[problem_type]
            for pipeline in pipelines:
                path_to_pipeline = path + 'primitives-interfaces\\jhu_primitives\\pipelines\\' + pipeline

                spec = importlib.util.spec_from_file_location(pipeline, path_to_pipeline + '.py')

                module = importlib.util.module_from_spec(spec)

                spec.loader.exec_module(module)

                pipeline_class = getattr(module, pipeline)

                pipeline_object = pipeline_class()

                pipeline_dir = dir(module)

                primitives = [prim for prim in primitive_names if prim in pipeline_dir]

                for dataset in datasets:

                    with open('temp.json', 'w') as file:
                        text = pipeline_object.get_json()
                        file.write(str(text))

                    json_object = json.load(open('temp.json', 'r'))
                    pipeline_id = json_object['id']

                    for primitive in primitives:
                        temp_path = jhu_path + d3m_string + primitive + '\\' + versions[primitive] + '\\pipelines\\'
                        temp_dir = os.listdir(temp_path)

                        for file in temp_dir:
                            temp_pipeline_id, file_type = file.split('.')
                            if file_type == 'meta':
                                temp_json = json.load(open(temp_path + temp_pipeline_id + '.meta', 'r'))
                                temp_dataset = temp_json['problem'].split('_problem')[0]
                                if temp_dataset == dataset:
                                    os.remove(temp_path + temp_pipeline_id + '.meta')
                                    os.remove(temp_path + temp_pipeline_id + '.json')

                        shutil.copy(path + 'temp.json', jhu_path + d3m_string 
                                        + primitive + "\\" + versions[primitive] + '\\pipelines\\'
                                        + pipeline_id + '.json')       # creates the pipeline json 
                        
                        write_meta(pipeline_id, dataset, temp_path + pipeline_id)          
                    break
                break
            break

def write_meta(pipeline_id, dataset, path, TRAIN_or_TEST = 'TRAIN'):
    meta = {}
    meta['problem'] = dataset + '_problem_' + TRAIN_or_TEST
    meta['train_inputs'] = [dataset + '_problem_' + 'TRAIN']
    meta['test_inputs'] = [dataset + '_problem_' + 'TEST']

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

def data_file_uri(abs_file_path = "", uri = "file", datasetDoc = False):
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
        s = s + "/" + data_dir + "/" + folder + "_dataset/datasetDoc.json"

    if uri == "file":
        return "file://localhost" + s
    else:
        return local_drive + ":" + s
