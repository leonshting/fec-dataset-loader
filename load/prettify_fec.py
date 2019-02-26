import os.path as osp
import pandas as pd
import json
import tqdm
import argparse
import cv2
from scipy.stats import mode

from kungfutils.io.file_utils import recursive_mkdir


def triplets_from_meta(meta_info_path, split='train'):
    """
    Make triplets for train and test
    :param meta_info_path: path to data set meta information
    :param split: 'train' | 'test'
    """
    triplet_data_fn = osp.join(
        meta_info_path, 'faceexp-comparison-data-{}-public.csv'.format(split)
    )
    triplet_data = pd.read_csv(triplet_data_fn, error_bad_lines=False, header=None)

    mapping_fn = osp.join(meta_info_path, 'loaded_images_meta.json')
    with open(mapping_fn, 'r') as f:
        images_mapping = json.load(f)

    mapping_table = pd.DataFrame.from_dict(images_mapping, orient='index').reset_index()
    mapping_table = mapping_table.copy()
    mapping_table[0] = mapping_table[0].apply(lambda x: x.split('/')[-1])
    mapping_table.columns = ['url', 'fname']

    triplet_data = pd.merge(triplet_data, left_on=0, right=mapping_table, right_on='url', suffixes=('', '_0'))
    triplet_data = pd.merge(triplet_data, left_on=5, right=mapping_table, right_on='url', suffixes=('', '_5'))
    triplet_data = pd.merge(triplet_data, left_on=5, right=mapping_table, right_on='url', suffixes=('', '_10'))

    label_med = mode(triplet_data[[17, 19, 21, 23, 25, 27]].values, axis=1, nan_policy='omit')[0]
    triplet_data = triplet_data[['fname', 1, 2, 3, 4, 'fname_5', 6, 7, 8, 9, 'fname_10', 11, 12, 13, 14, 15]]

    triplet_data['label'] = triplet_data[15]
    triplet_data['label_triplet'] = label_med

    return triplet_data


def make_crops_from_triplets(meta_info_path, images_path, triplet_path, triplet_data, split='train'):
    """
    Make crops
    :param meta_info_path: path with meta info
    :param images_path: path with original images
    :param triplet_path: path to save cropped images
    :param triplet_data: data frame returned with the function above
    :param split: 'train' | 'test'
    """

    triplet_path = osp.join(triplet_path, split)
    recursive_mkdir(triplet_path)

    data_set = []
    for num, row in tqdm.tqdm(triplet_data.iterrows()):
        ims = [cv2.imread(osp.join(images_path, row[i])) for i in ['fname', 'fname_5', 'fname_10']]
        boxes = [[row[i + j] for j in range(1, 5)] for i in [0, 5, 10]]

        faces = []
        for im, b in zip(ims, boxes):
            h, w = im.shape[:2]
            faces.append(im[int(b[2] * h):int(b[3] * h), int(b[0] * w):int(b[1] * w)])

        per_tr_data = []
        for face_num, face in enumerate(faces):
            fn = osp.join(triplet_path, 'triplet_{}_{}.jpg'.format("%09d" % num, face_num))
            cv2.imwrite(fn, face)
            per_tr_data.append(fn)

        per_tr_data.append(row['label'])
        per_tr_data.append(row['label_triplet'])

        data_set.append(per_tr_data)
    data_set_df = pd.DataFrame(data_set, columns=['im1', 'im2', 'im3', 'label', 'label_triplet'])

    data_set_df.im1 = data_set_df.im1.apply(lambda x: x.split('/')[-1])
    data_set_df.im2 = data_set_df.im2.apply(lambda x: x.split('/')[-1])
    data_set_df.im3 = data_set_df.im3.apply(lambda x: x.split('/')[-1])

    meta_final_fn = osp.join(meta_info_path, '{}_triplets_info.csv'.format(split))
    data_set_df.to_csv(meta_final_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_meta', help='folder path with original zip contents', type=str)
    parser.add_argument('--path_to_images', help='path with image save folder', type=str, default='~/fec_images')
    parser.add_argument('--path_to_triplets', help='path to save crops', type=str, default='~/fec_triplets')
    parser.add_argument('--split', help='train or test split', type=str, default='train')

    args = parser.parse_args()

    triplets = triplets_from_meta(meta_info_path=args.path_to_meta, split=args.split)
    make_crops_from_triplets(
        meta_info_path=args.path_to_meta, images_path=args.path_to_images,
        triplet_path=args.path_to_triplets, triplet_data=triplets, split=args.split)

