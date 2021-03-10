#! /usr/bin/env python

# util.py
# Created on 2017-09-14.

import os
import argparse
import importlib
import importlib.util
import json
import shutil
import re

PROBLEM_TYPES = [
     "matching",
     "graphMatching",
     "vertexClassification",
     "communityDetection",
     "linkPrediction",
    ]

DATASETS = {
             "matching": [
                 "datasets/seed_datasets_current/LL1_2734_CLIR",
                 "datasets/seed_datasets_current/LL1_5297_CLIR2",
                 ],
             "graphMatching": [
                 "datasets/seed_datasets_current/49_facebook_MIN_METADATA",
                 "datasets/seed_datasets_current/LL1_DIC28_net_MIN_METADATA",
                 ],
            "vertexClassification": [
                "datasets/seed_datasets_current/LL1_net_nomination_seed_MIN_METADATA",
                "datasets/seed_datasets_current/LL1_EDGELIST_net_nomination_seed_MIN_METADATA",
                "datasets/seed_datasets_current/LL1_VTXC_1343_cora_MIN_METADATA",
                "datasets/seed_datasets_current/LL1_VTXC_1369_synthetic_MIN_METADATA",
                ],
              "communityDetection": [
                  "datasets/seed_datasets_current/LL1_bn_fly_drosophila_medulla_net_MIN_METADATA",
                  ],
             "linkPrediction": [
                 "datasets/seed_datasets_current/59_umls_MIN_METADATA",
                 "datasets/seed_datasets_current/59_LP_karate_MIN_METADATA"
                 ]
            }

PIPELINES = {
             "graphMatching": [
                 "sgm_pipeline",
            #     "sgm_pipeline_10"
                 ],
            "vertexClassification": [
                "gclass_ase_pipeline",
                "gclass_lse_pipeline",
                # "gclass_oosase_pipeline",
                # "gclass_ooslse_pipeline",
                # "sgc_pipeline"
                ],
              "communityDetection": [
                 "gmm_ase_pipeline",
                 "gmm_lse_pipeline",
            #     "gmm_oosase_pipeline",
            #     "gmm_ooslse_pipeline"
            #     "sgc_pipeline"
                 ],
             "linkPrediction": [
                 "link_pred_pipeline",
                 ],
              "matching": [
                 "euclidean_nomination_pipeline",
                 "procrustes_nomination_pipeline",
                 "nearest_neighbor_nomination_pipeline",
                 "sgm_nomination_pipeline",
                 "asgm_nomination_pipeline"
                 ],
             }

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def load_args():
    parser = argparse.ArgumentParser(description = "Pipeline runner")

    parser.add_argument(
        'target_repo',
        action = 'store',
        help = "the type of object to generate jsons for",
    )

    parser.add_argument(
        'problem_type',
        action = 'store',
        help = 'the d3m problem type to check'
    )
    # parser.add_argument(
    #     'primitives_or_pipelines',
    #     action = 'store',
    #     help = "the type of object to generate jsons for",
    # )
    arguments = parser.parse_args()
    return [arguments.target_repo, arguments.problem_type]

def generate_json(target_repo, type_):
    if type_ not in ['pipelines', 'primitives']:
        raise ValueError("Unsupported object type; 'pipelines' or 'primitives' only.")

    VERSION = "primitives"
    path = os.path.join(os.path.abspath(os.getcwd()),"")

    jhu_path = os.path.join(path, target_repo, VERSION, "JHU", "")

    all_primitives = os.listdir(jhu_path)
    primitive_names = [primitive.split('.')[-2] for primitive in all_primitives]

    versions = {}
    for i in range(len(all_primitives)):
        versions[primitive_names[i]] = os.listdir(os.path.join(jhu_path, all_primitives[i]))[0]

    print()
    print(versions)
    print()

    if type_ == 'primitives':
        for i in range(len(all_primitives)):
            temp = jhu_path + all_primitives[i]
            print()
            print(temp)
            print()
            os.system('python3 -m d3m index describe -i 4 ' + all_primitives[i] + ' > ' + os.path.join(temp, versions[primitive_names[i]], 'primitive.json'))
    else:
        python_paths = {}
        for python_path in all_primitives:
            temp_path = os.path.join(jhu_path, python_path, versions[python_path.split('.')[-2]], 'pipelines', "")
            temp_dir = os.listdir(temp_path)
            python_paths[python_path.split('.')[-2]] = python_path

            for file in temp_dir:
                os.remove(temp_path + file)
            temp_path = os.path.join(jhu_path, python_path, versions[python_path.split('.')[-2]], 'pipeline_runs', "")
            for file in os.listdir(temp_path):
                os.remove(temp_path + file)

        paths_to_pipelines = {}
        for problem_type in PROBLEM_TYPES:
            datasets = DATASETS[problem_type]
            pipelines = PIPELINES[problem_type]
            paths_to_pipelines[problem_type] = []
            for pipeline in pipelines:

                print()
                print(pipelines)
                print()

                path_to_pipeline = os.path.join(path, 'primitives-interfaces', 'jhu_primitives', 'pipelines', pipeline)

                spec = importlib.util.spec_from_file_location(pipeline, path_to_pipeline + '.py')

                module = importlib.util.module_from_spec(spec)

                spec.loader.exec_module(module)

                pipeline_class = getattr(module, pipeline)

                pipeline_dir = dir(module)
                p_dir = [convert(p) for p in pipeline_dir]
                primitives = [prim for prim in primitive_names if prim in p_dir]

                for dataset in datasets:
                    dataset_name = dataset.split("/")[-1]
                    pipeline_object = pipeline_class()
                    dataset_new = dataset_name + '_dataset'

                    with open('temp.json', 'w') as file:
                        text = pipeline_object.get_json()
                        file.write(str(text))

                    json_object = json.load(open('temp.json', 'r'))
                    pipeline_id = json_object['id']
                    primitive_id = json_object['steps'][-1]['primitive']['id']

                    for primitive in primitives:
                        temp_path = os.path.join(jhu_path, python_paths[primitive], versions[primitive], 'pipelines', "")
                        full_path = os.path.join(jhu_path, python_paths[primitive], versions[primitive], 'pipelines', pipeline_id)
                        shutil.copy(os.path.join(path, 'temp.json'),
                                    full_path + '.json')       # creates the pipeline json
                        #write_meta(dataset_name, dataset_new, temp_path + pipeline_id)
                        write_pipeline_run(dataset, full_path)
                        paths_to_pipelines[problem_type].append(full_path)
        os.remove('temp.json')
        return paths_to_pipelines

def write_meta(dataset_name, dataset_new, path):
    meta = {}
    meta['problem'] = dataset_name + '_problem'
    meta['full_inputs'] = [dataset_new]
    meta['train_inputs'] = [dataset_new + "_TRAIN"]
    meta['test_inputs'] = [dataset_new + "_TEST"]
    meta['score_inputs'] = [dataset_new + "_SCORE"]
    with open(path + '.meta', 'w') as file:
        json.dump(meta, file)


def pipeline_run(problem_type, target_repo, paths_to_pipelines):
    datasets = DATASETS[problem_type]
    paths_to_pipelines_problem_type = paths_to_pipelines[problem_type]

    unique_paths = []
    unique_ids = []
    for path in paths_to_pipelines_problem_type:
        pipeline_id = path.split("/")[-1]
        if pipeline_id not in unique_ids:
            unique_paths.append(path)
            unique_ids.append(pipeline_id)

    for dataset in datasets:
        for path in unique_paths:
            dataset_path = dataset + "/"

            print()
            print(path)
            print()

            cmd = "python3 -m d3m runtime fit-score -p " + path + ".json"
            cmd += " -r " + dataset_path + dataset.split('/')[-1] + "_problem/problemDoc.json"
            cmd += " -i " + dataset_path + "TRAIN/dataset_TRAIN/datasetDoc.json"
            cmd += " -t " + dataset_path + "TEST/dataset_TEST/datasetDoc.json"
            cmd += " -a " + dataset_path + "SCORE/dataset_SCORE/datasetDoc.json"
            run_path = '/'.join(path.split('/')[:-2]) + "/pipeline_runs/" + path.split('/')[-1]
            cmd += " -O " + run_path + "_pipeline_run.yml"
            #print(cmd, file=sys.stderr)
            os.system(cmd)

def write_pipeline_run(dataset, path):
    dataset_path = dataset + "/"
    cmd = "python3 -m d3m runtime fit-score -p " + path + ".json"
    cmd += " -r " + dataset_path + dataset.split('/')[-1] + "_problem/problemDoc.json"
    cmd += " -i " + dataset_path + "TRAIN/dataset_TRAIN/datasetDoc.json"
    cmd += " -t " + dataset_path + "TEST/dataset_TEST/datasetDoc.json"
    cmd += " -a " + dataset_path + "SCORE/dataset_SCORE/datasetDoc.json"
    run_path = '/'.join(path.split('/')[:-2]) + "/pipeline_runs/"
    run_path += path.split('/')[-1] + "-" + dataset.split('/')[-1] + "_pipeline_run.yml"
    cmd += " -O " + run_path
    # print(cmd, file=sys.stderr)
    os.system(cmd)
    os.system("gzip " + run_path)

def pipeline_run_all(paths_to_pipelines):
    for problem_type in PROBLEM_TYPES:
        paths_to_pipelines_problem_type = paths_to_pipelines[problem_type]
        for path in paths_to_pipelines_problem_type:
            run_path = '/'.join(path.split('/')[:-2]) + "/pipeline_runs/"
            for file in os.listdir(run_path):
                os.remove(run_path + file)
    for problem_type in PROBLEM_TYPES:
        paths_to_pipelines_problem_type = paths_to_pipelines[problem_type]
        # print(paths_to_pipelines_problem_type, file=sys.stderr)
        datasets = DATASETS[problem_type]
        for dataset in datasets:
            for path in paths_to_pipelines_problem_type:
                dataset_path = dataset + "/"
                pipeline_dataset = path.split('/')[-1] + '-' + dataset.split('/')[2]
                problem_path = '/'.join(path.split('/')[:-2]) + "/pipeline_runs/" + path.split('/')[-1]
                cmd = "python3 -m d3m runtime fit-score"
                cmd += " -p " + path + ".json" # pipelines/id.json
                cmd += " -r " + dataset_path + dataset.split('/')[-1] + "_problem/problemDoc.json"
                cmd += " -i " + dataset_path + "TRAIN/dataset_TRAIN/datasetDoc.json"
                cmd += " -t " + dataset_path + "TEST/dataset_TEST/datasetDoc.json"
                cmd += " -a " + dataset_path + "SCORE/dataset_SCORE/datasetDoc.json"
                cmd += " -O " + problem_path + "_pipeline_run.yml" # pipeline_runs/id.yml
                # print(cmd, file=sys.stderr)
                os.system(cmd)

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
        s = s + "/" + data_dir
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


if __name__ == '__main__':
    target_repo, problem_type = load_args()
    generate_json(target_repo, "primitives")
    paths_to_pipelines = generate_json(target_repo, "pipelines")
    # target_repo, type_ = load_args()
    # generate_json(target_repo, type_)

    #if problem_type == 'all':
    #    pipeline_run_all(paths_to_pipelines)
    #else:
    #    pipeline_run(problem_type, target_repo, paths_to_pipelines)
