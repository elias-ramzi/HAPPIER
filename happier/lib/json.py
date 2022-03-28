import json


def save_json(obj, path):
    with open(path, 'w') as fp:
        json.dump(obj, fp)


def load_json(path):
    with open(path) as fp:
        db = json.load(fp)
    return db
