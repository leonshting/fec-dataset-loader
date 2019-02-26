### This package loads images and composes these images into triplets according to Google [facial expression comparison dataset](https://ai.google/tools/datasets/google-facial-expression/)

0. Fetch submodule with some useful utils `git pull --recurse-submodules` 

1. The package uses Tor service to rotate IPs since it was observed, that connection to Flickr,
 where most of photos from dataset are hosted, stalls after some number of downloads. 
 So the first stage is to start tor service, with which IPs are rotated, refer to this beautifully useful [repo](https://github.com/alex-miller-0/Tor_Crawler)

2. Once the service is started and dataset meta is loaded from Google servers, make sure you `pip install -r requirements.txt`
3. The first actual dataset related command is the following: `python3 ./load/load_fec.py --tor_pass=mypass --path_to_meta=<meta_info_path> --path_to_images=<path_to_save_images>`, this will take some time, depends on your bandwidth
4. The second command is: `python3 ./load/prettify_fec.py --path_to_meta=<meta_info_path> --path_to_images=<path_with_saved_images> --path_to_triplets=/home/mininlab/DATA/fec_try_triplets --split=(train|test)`

At the end you will have 3 new files in `<meta_info_path>`: list with loaded and missed images (`loaded_images_meta.json`, `missed_images_meta`) and final markup file (`(train|test)_triplets_info.csv`) and 
2 folders with genuine images and triplet images respectively.