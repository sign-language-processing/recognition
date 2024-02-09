from functools import lru_cache

import numpy as np
import json

from pose_format import Pose
from huggingface_hub import hf_hub_download

try:
    import tflite_runtime.interpreter as tflite
except (ImportError, ModuleNotFoundError):
    import tensorflow as tf

    tflite = tf.lite

HUGGINGFACE_REPO_ID = "sign/kaggle-asl-signs-1st-place"


@lru_cache(maxsize=1)
def get_paths():
    model_path = hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename="model.tflite")
    index_map_path = hf_hub_download(repo_id=HUGGINGFACE_REPO_ID, filename="sign_to_prediction_index_map.json")
    return {
        "model": model_path,
        "index_map": index_map_path
    }


@lru_cache(maxsize=1)
def get_model_runner():
    paths = get_paths()
    model = tflite.Interpreter(paths["model"])
    return model.get_signature_runner('serving_default')


def prepare_pose(pose: Pose):
    # Reorder the pose components based on the Kaggle implementation
    pose = pose.get_components([
        "FACE_LANDMARKS",
        "LEFT_HAND_LANDMARKS",
        "POSE_LANDMARKS",
        "RIGHT_HAND_LANDMARKS"
    ])

    # scale the point values
    pose.body.data = np.float32(pose.body.data / np.array(
        [pose.header.dimensions.width, pose.header.dimensions.height, 1]))

    # normalize the pose
    info = pose.header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                          p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))
    pose = pose.normalize(info)

    return pose


@lru_cache(maxsize=1)
def get_labels():
    paths = get_paths()
    with open(paths["index_map"], 'r', encoding="utf-8") as f:
        index_map = json.load(f)

    # Invert the index map
    return {v: k for k, v in index_map.items()}


def prob_to_label(prob):
    labels = get_labels()
    return labels[np.argmax(prob)]


def predict(pose: Pose, label=False):
    prepared_pose = prepare_pose(pose)
    tensor = prepared_pose.body.data.filled(0)
    prediction_fn = get_model_runner()

    output = prediction_fn(inputs=tensor)
    class_prob = output['outputs'].reshape(-1)

    if label:
        return prob_to_label(class_prob)

    return class_prob
