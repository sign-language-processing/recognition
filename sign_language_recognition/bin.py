#!/usr/bin/env python

import argparse
import importlib
import math

from pose_format.pose import Pose
from tqdm import tqdm
import pympi


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        choices=['kaggle_asl_signs'],
                        help='model to use')
    parser.add_argument('--pose', required=True, type=str, help='path to input pose file')
    parser.add_argument('--elan', required=True, type=str, help='path to elan file')

    return parser.parse_args()


def main():
    args = get_args()

    module = importlib.import_module(f"sign_language_recognition.{args.model}")

    print('Loading input pose...')
    with open(args.pose, 'rb') as pose_file:
        pose = Pose.read(pose_file.read())

    print('Loading ELAN file...')
    eaf = pympi.Elan.Eaf(file_path=args.elan, author="sign-langauge-processing/recognition")
    sign_annotations = eaf.get_annotation_data_for_tier('SIGN')

    print('Predicting signs...')
    for segment in tqdm(sign_annotations):
        start_frame = int((segment[0] / 1000) * pose.body.fps)
        end_frame = math.ceil((segment[1] / 1000) * pose.body.fps)

        cropped_pose = Pose(
            header=pose.header,
            body=pose.body[start_frame:end_frame]
        )
        gloss = module.predict(cropped_pose, label=True)
        eaf.remove_annotation('SIGN', segment[0])
        eaf.add_annotation('SIGN', segment[0], segment[1], gloss)

    print('Saving ELAN file...')
    eaf.to_file(args.elan)


if __name__ == '__main__':
    main()
    # python -m sign_language_recognition.bin --model="kaggle_asl_signs" --pose="sign.pose" --elan="sign.eaf"
