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

if __name__ == '__main__':
    generate_json()

