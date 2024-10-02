import os.path as osp
import sys
import pickle

with open(r"/home/dojan/FOCUS-1/SelfPose3d/output/output.pkl", "rb") as f:
    output = pickle.load(f)

preds = output["preds"]
preds_2d = output["preds_2d"]
roots = output["roots"]

