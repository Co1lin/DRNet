import json
import pickle

def read_json_file(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return out

def read_pkl_file(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out