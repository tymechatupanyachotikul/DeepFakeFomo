import os
import argparse
import numpy as np
from tqdm import tqdm

def ANNList_Generation(output_dir, output_name):
    # Update your data path below
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            pass



if __name__ == '__main__':
    # generate ann file for training/testing
    parser = argparse.ArgumentParser(
        description='''generate ann file list for training/testing''')
    parser.add_argument('--output_dir', default='./output', type=str,
                        help='directory of stored LaRE map')
    parser.add_argument('--output_name', default='', type=str,
                        help='name of output ann file')

    args = parser.parse_args()

    ANNList_Generation(args.output_dir, args.output_name)
