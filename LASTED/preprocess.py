import os
import tarfile
import argparse
import numpy as np
from tqdm import tqdm

data_root = 'GenImage/'
save_path = 'annotation/'

def FileList_Generation(dataset_file, dataset_name, split, reconstructed, reconstruct_only):

    tar = tarfile.open(args.dataset_file, 'r:gz')
    members = tar.getmembers()

    if reconstruct_only:
        data_name_list = [
            (f'./{dataset_name}/{split}/nature_reconstruct', 1)
        ]
    else:
        data_name_list = [
            # e.g., (path_to_dataset, label)
            (f'./{dataset_name}/{split}/nature_reconstruct', 1) if reconstructed else (f'./{dataset_name}/{split}/ai', 1),  # Fake Dataset
            (f'./{dataset_name}/{split}/nature', 0),  # Real Dataset
        ]

    img_list = []
    for data_name, label in data_name_list:
        if label == 0:
            continue
        path = '%s/' % data_name
        file_list = [member.name for member in members if member.name.startswith(path) and member.isfile()]
        file_list = sorted(file_list)
        for file in tqdm(file_list):
            img_list.append((file, label))

    img_list2 = []
    for data_name, label in data_name_list:
        if label == 1:
            continue
        path = '%s/' % data_name
        file_list = [member.name for member in members if member.name.startswith(path) and member.isfile()]
        np.random.shuffle(file_list)
        file_list = file_list[:len(img_list)]
        for file in tqdm(file_list):
            img_list2.append((file, label))
    
    np.random.shuffle(img_list2)
    img_list2 = img_list2[:len(img_list)]
    img_list = img_list + img_list2

    print('#Images: %d' % len(img_list))
    
    if reconstruct_only:
        textfile = open(f'{save_path}{split}_{dataset_name}_reconstruct_only.txt', 'w')
    else:
        textfile = open(f'{save_path}{split}_{dataset_name}{"_reconstructed" if reconstructed else ""}.txt', 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()


if __name__ == '__main__':
    # generate file list for training/testing
    # E.g., Train_VISION.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    parser = argparse.ArgumentParser(description='''generate file list for training/testing''')
    
    parser.add_argument('--dataset_file', default='', type=str, help='.tar.gz dataset file path')
    parser.add_argument('--dataset_name', default='', type=str, help='name of dataset')
    parser.add_argument('--split', default='', type=str, help='dataset split')
    parser.add_argument('--reconstructed', action='store_true', help='Use reconstructed dataset as fake images')
    parser.add_argument('--reconstruct_only', action='store_true', help='Extract only reconstructed images')

    args = parser.parse_args()

    FileList_Generation(args.dataset_file, args.dataset_name, args.split, args.reconstructed, args.reconstruct_only)