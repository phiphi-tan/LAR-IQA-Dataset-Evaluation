from dataset_loader import *
from findings import dataset_quality
from pathlib import Path

import cv2
import os
import numpy as np

def average(lst):
    if len(lst) == 0:
        return 0
    
    return sum(lst) / len(lst)

dataset_imgs = {
    # "tallyqa_simple_256":tally_qa(split_size=256, is_complex=False),
    # "tallyqa_complex_256":tally_qa(split_size=256, is_complex=True),
    "tallyqa_256":tally_qa(split_size=256, is_complex=None),
    # "text_ocr_256":text_ocr(split_size=256),
    # "ocr_vqa_256":ocr_vqa(split_size=256),
    # "drone_detection_64":drone_detection(split_size=64),
    # "drone_detection_256":drone_detection(split_size=256),
    # "weapon_detection_64":weapon_detection(split_size=64),
    # "flickr8k_64":flickr8k(split_size=64),
    # "flickr8k_256":flickr8k(split_size=256),
    # "flickr8k_128":flickr8k(split_size=128),
}

quality_boundaries = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

OUTPUT_DIR = Path("outputs/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # create output directory

def create_directories(dir_path, boundaries):
    first_dir_name = f"0-{boundaries[0]}"
    first_dir = Path(os.path.join(dir_path, first_dir_name))
    first_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(boundaries) - 1):
        range_dir_name = f"{boundaries[i]}-{boundaries[i+1]}"
        range_dir = Path(os.path.join(dir_path, range_dir_name))
        range_dir.mkdir(parents=True, exist_ok=True)

    last_dir_name = f"{boundaries[-1]}-1"
    last_dir = Path(os.path.join(dir_path, last_dir_name))
    last_dir.mkdir(parents=True, exist_ok=True)

    main_dir =  Path(os.path.join(dir_path,"all"))
    main_dir.mkdir(parents=True, exist_ok=True)


    return

def get_directory(dir_path, score, boundaries):
    if score < boundaries[0]:
        return Path(os.path.join(dir_path, f"0-{boundaries[0]}"))
    
    for i in range(1, len(boundaries)):
        if score < boundaries[i]:
            return Path(os.path.join(dir_path, f"{boundaries[i-1]}-{boundaries[i]}"))

    return  Path(os.path.join(dir_path,f"{boundaries[-1]}-100"))


def addText(img, text):
    new_img = img.copy()

    new_img = cv2.putText(
            new_img, # image on which to draw text
            text, 
        (50,50), # bottom left corner of text
        cv2.FONT_HERSHEY_SIMPLEX, # font to use
        1, # font scale
        (255, 200, 200), # color
        2, # line thickness
    )
    return new_img

for dataset, qualities in dataset_quality.items():
    try:
        print('average quality of {} is:{}'.format(dataset, round(average(qualities), 3)
))    
        dataset_dir = Path(os.path.join(OUTPUT_DIR, dataset))
        dataset_dir.mkdir(parents=True, exist_ok=True) # create individual dataset directories

        # make directory for each quality boundary
        create_directories(dir_path=dataset_dir, boundaries=quality_boundaries)

        for i in range(len(qualities)):
            print('sorting image {} / {}'.format(i, len(qualities)))
            curr_quality = qualities[i]
            curr_img = dataset_imgs[dataset][i].copy() 
            curr_img = cv2.cvtColor(np.array(curr_img), cv2.COLOR_RGB2BGR)
            main_dir =  Path(os.path.join(dataset_dir,"all"))

            cv2.imwrite(os.path.join(main_dir, "{}.jpg".format(i)), curr_img)

            curr_img = addText(curr_img, str(round(curr_quality, 3)))

            img_path = get_directory(dir_path=dataset_dir, score=curr_quality, boundaries=quality_boundaries)
            cv2.imwrite(os.path.join(img_path, "{}.jpg".format(i)), curr_img)
    except KeyError:
        print("Corresponding image data for {} not found".format(dataset))
        continue





