import pandas as pd
import os.path as osp
import logging
import tqdm
import json
import argparse
import requests

from PIL import Image
import io
import sys

from kungfutils.io.file_utils import recursive_mkdir
from kungfutils.routines.string_routines import generate_id
from tor_crawler import TorCrawler


class FECLoader:
    def __init__(self, path_to_zip_contents, path_to_images='~/fec_images', logger=None):
        if logger is None:
            handler = logging.FileHandler(osp.join(path_to_zip_contents, 'fec_load_{}'.format(generate_id())))
            console_handler = logging.StreamHandler(sys.stdout)

            self._logger = logging.getLogger('fec_loader')
            self._logger.addHandler(handler)
            self._logger.addHandler(console_handler)

            self._logger.setLevel(logging.INFO)
        else:
            self._logger = logger

        self._path_to_contents = path_to_zip_contents

        image_meta = {}
        for split in ['train', 'test']:
            split_path = osp.join(self._path_to_contents, 'faceexp-comparison-data-{}-public.csv'.format(split))
            image_meta[split] = pd.read_csv(split_path, error_bad_lines=False, header=None)

        self._image_root = path_to_images
        recursive_mkdir(self._image_root)

        pd.concat([split[col_id] for col_id in [0, 5, 10] for split in image_meta.values()])

        df_concat = pd.concat(list(image_meta.values()))
        self._images_unq = pd.concat([df_concat[i] for i in [0, 5, 10]]).unique()

        self._logger.info('Images meta information is OK!')

    def load(self, crawler=None):
        if crawler is None:
            self._logger.info('Using requests to fetch images, this is not tested')
        crawler = crawler or requests

        url_to_fname = {}
        missed_url = {}
        for num, url in tqdm.tqdm(enumerate(self._images_unq)):
            ext = url.split('.')[-1]
            fname = ".".join([osp.join(self._image_root, "%010d" % num), ext])
            try:
                res = crawler.get(url=url)
                if res.status_code != 200:
                    self._logger.info("on {}: error code: {}".format(url, res.status_code))
                    raise ConnectionError
                Image.open(io.BytesIO(res.content)).save(fname)
                url_to_fname[url] = fname
            except ConnectionError:
                missed_url_map[url] = fname

        return url_to_fname, missed_url

    def load_with_crawler(self, tor_pass):
        crawler = TorCrawler(ctrl_pass=tor_pass, use_bs=False)
        return self.load(crawler)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tor_pass', help='password to tor service', type=str)
    parser.add_argument('--path_to_meta', help='folder path with original zip contents', type=str)
    parser.add_argument('--path_to_images', help='path to image save folder', type=str, default='~/fec_images')

    args = parser.parse_args()

    loader = FECLoader(path_to_images=args.path_to_images, path_to_zip_contents=args.path_to_meta)
    url_to_fname_map, missed_url_map = loader.load_with_crawler(tor_pass=args.tor_pass)

    with open(osp.join(args.path_to_meta, 'loaded_images_meta.json'), 'w') as f:
        json.dump(fp=f, obj=url_to_fname_map)

    with open(osp.join(args.path_to_meta, 'missed_images_meta.json'), 'w') as f:
        json.dump(fp=f, obj=missed_url_map)

