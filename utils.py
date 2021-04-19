from os import path
from skimage import io
import numpy as np
import cv2
import os
import json


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)
    return img


def write_text_detection_results(base_file_name, img, text_detection_results):

    path = os.path.dirname(os.path.realpath(__file__))
    dirname = os.path.join(path, base_file_name, 'results')
    os.makedirs(dirname, exist_ok=True)
    reg_score_img_file = os.path.join(dirname, "region_score" + '.jpg')
    cv2.imwrite(reg_score_img_file, text_detection_results['region_score_map'])

    affinity_score_img_file = os.path.join(dirname, "affinity_score" + '.jpg')
    cv2.imwrite(affinity_score_img_file, text_detection_results['affinity_score_map'])

    cc_mask_img_file = os.path.join(dirname, "mask" + '.jpg')
    cv2.imwrite(cc_mask_img_file, text_detection_results['connected_component_mask'])


    bbox_file_path = os.path.join(dirname, 'bboxes.txt')

    for k, v in text_detection_results['bboxes'].items():
        crop_image_abs_path = os.path.join(dirname, k + '.jpg')
        bbox_coords = v.split(',')
        min_x = (int)(bbox_coords[0])
        min_y = (int)(bbox_coords[1])
        max_x = (int)(bbox_coords[4])
        max_y = (int)(bbox_coords[5])
        crop = img[min_y:max_y, min_x:max_x, :]
        cv2.imwrite(crop_image_abs_path, crop)


def write_recognition_results(base_file_name, recognition_results):
    # save fiducial to image
    path = os.path.dirname(os.path.realpath(__file__))
    dirname = os.path.join(path, base_file_name, 'results')
    os.makedirs(dirname, exist_ok=True)
    rec_results = {}
    for k, v in recognition_results.items():
        rec_result = v['result']
        rec_results[k] = rec_result
        stn_input_img_path = os.path.join(dirname, "fid_" + k + '.jpg')
        stn_input_with_fid = v['stn_input_with_fid']
        stn_input_with_fid.save(stn_input_img_path)
        stn_output = v['stn_output']
        stn_out_img_path = os.path.join(dirname, "stn_output_" + k + '.jpg')
        stn_output.save(stn_out_img_path)
    with open(os.path.join(dirname, "rec_results.txt"), 'w') as f:
        json.dump(rec_results, f)
