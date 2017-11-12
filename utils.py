# coding=utf-8
import os

PROJECT_NAME = 'KerasSamples'


def check_file(file_path):
    if not os.path.exists(file_path):
        raise Exception('File "%s" not found.' % file_path)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def root_path(path):
    root = os.path.abspath('.')
    while not root.endswith(PROJECT_NAME):
        root = os.path.dirname(root)
    for layer in path.split('/'):
        root = os.path.join(root, layer)
    return root


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count
