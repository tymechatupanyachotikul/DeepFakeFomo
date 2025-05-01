import os
import argparse
import numpy as np
from tqdm import tqdm

data_root = 'GenImage/'
save_path = 'annotation/'

def FileList_Generation(dataset_name, split, reconstructed, reconstruct_only):
    # Update your data path below
    if reconstruct_only:
        data_name_list = [
            (f'{dataset_name}/{split}/nature_reconstruct', 1)
        ]
    else:
        data_name_list = [
            # e.g., (path_to_dataset, label)
            (f'{dataset_name}/{split}/nature_reconstruct', 1) if reconstructed else (f'{dataset_name}/{split}/ai', 1),  # Fake Dataset
            (f'{dataset_name}/{split}/nature', 0),  # Real Dataset
        ]

    img_list = []
    for data_name, label in data_name_list:
        if label == 0:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        for file in tqdm(flist):
            img_list.append((data_root + path + file, label))
    img_list2 = []
    for data_name, label in data_name_list:
        if label == 1:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        np.random.shuffle(flist)
        flist = flist[:len(img_list)]
        for file in tqdm(flist):
            img_list2.append((data_root + path + file, label))
    np.random.shuffle(img_list2)
    img_list2 = img_list2[:len(img_list)]
    img_list = img_list + img_list2

    print('#Images: %d' % len(img_list))
    if reconstructed:
        textfile = open(f'{save_path}reconstruct_{dataset_name}.txt', 'w')
    else:
        textfile = open(f'{save_path}{split}_{dataset_name}{"_reconstructed" if reconstructed else ""}.txt', 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()


if __name__ == '__main__':
    # generate file list for training/testing
    # E.g., Train_VISION.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    parser = argparse.ArgumentParser(
        description='''generate file list for training/testing''')
    parser.add_argument('--dataset_name', default='', type=str,
                        help='name of dataset')
    parser.add_argument('--split', default='', type=str,
                        help='dataset split')
    parser.add_argument('--reconstructed', default=True, type=str,
                        help='use reconstructed dataset as fake images')
    parser.add_argument('--reconstruct_only', default=False, type=str,
                        help='extract only reconstructed images')

    args = parser.parse_args()

    FileList_Generation(args.dataset_name, args.split, args.reconstructed)
