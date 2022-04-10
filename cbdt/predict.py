#!/usr/bin/python3
""" Prediction script for use with TIRA"""
import json
import pickle

from cbdt.features.cbmodel import ClickbaitModel
from cbdt.features.dataset import ClickbaitDataset

if __name__ == '__main__':
    dataset = ClickbaitDataset(
        instances_path="../corpus/clickbait17-train-170331/instances.jsonl",
        truth_path="../corpus/clickbait17-train-170331/truth.jsonl"
    )

    f_builder = pickle.load(open("feature_builder.pkl", "rb"))
    f_builder.build(dataset)
    x = f_builder.build_features

    cbm = ClickbaitModel()
    cbm.load("model_trained.pkl")

    y = cbm.predict(x)

    id_list = sorted(dataset.dataset_dict.keys())
    _results_list = []
    for i in range(len(id_list)):
        _results_list.append({'id': id_list[i], 'clickbaitScore': y[i]})

    with open("../corpus/clickbait17-train-170331/result.jsonl", "w") as of:
        for line in _results_list:
            line['clickbaitScore'] = int(line['clickbaitScore'])
            of.write(json.dumps(line))
            of.write("\n")
