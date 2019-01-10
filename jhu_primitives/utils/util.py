#! /usr/bin/env python

# util.py
# Created on 2017-09-14.

import os
#import d3m.index

def file_path_conversion(abs_file_path, uri="file"):
    local_drive= abs_file_path.split(':')[0]

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

def generate_json():
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

    path = os.path.abspath(os.getcwd()) + "\\primitives_repo\\" + version + "\\JHU\\"

    for i in os.listdir(path):
        temp = path + i
        for j in os.listdir(temp): 
            os.system('python -m d3m.index describe -i 4 ' + i + ' > ' + temp + '\\' + j + '\\primitive.json')

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

if __name__ == '__main__':
    generate_json()

