import tensorflow as tf
import numpy as np
from pymongo import MongoClient
from pprint import pprint


mongo = MongoClient()
db = mongo.albart
tracks = db.tracks


def json_to_feature(json):
    feature = []
    analysis = json['analysis']
    analysis_keys = analysis.keys()
    analysis_keys.sort()
    for key in analysis_keys:
        if analysis[key] is not None:
            feature.append(analysis[key])
    return np.array(feature)


for track in tracks.find():
    pprint(track)
    print(json_to_feature(track))
    break
