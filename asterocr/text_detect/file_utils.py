# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from .imgproc import *

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

# borrowed: https://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in walklevel(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def saveResult(img_file, img, boxes, score_text, score_link, cc_mask, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))
        dirname = os.path.dirname(img_file) + '/results/' + filename
        # result directory
        os.makedirs(dirname, exist_ok=True)

        bbox_file_path = os.path.join(dirname, 'bboxes.txt')
        # The image that shows the bounding boxes for detected text regions drawn on the original image
        res_img_path = os.path.join(dirname, "res_" + filename + '.jpg')

        # save the region score, affinity score and connected component images
        # convert to heatmap:
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        # region score
        region_score_hm = imgproc.cvt2HeatmapImg(score_text)
        reg_score_img_file = os.path.join(dirname, "reg_" + filename + '.jpg')
        cv2.imwrite(reg_score_img_file, region_score_hm)
        # affinity score
        affinity_score_hm = imgproc.cvt2HeatmapImg(score_link)
        affinity_score_img_file = os.path.join(dirname, "affinity_" + filename + '.jpg')
        cv2.imwrite(affinity_score_img_file, affinity_score_hm)

        # connected components
        cc_mask_img_file = os.path.join(dirname, "cc_mask_" + filename + '.jpg')
        cc_mask_hm = imgproc.cvt2HeatmapImg(cc_mask)
        cv2.imwrite(cc_mask_img_file, cc_mask_hm)

        with open(bbox_file_path, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                x_coords = poly[0:-1:2]
                y_coords = poly[1::2]
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)
                crop = img[min_y:max_y, min_x:max_x, :]
                crop_img_file = "crop{0}".format(i) + '.jpg'
                crop_image_abs_path = os.path.join(dirname, crop_img_file)
                cv2.imwrite(crop_image_abs_path, crop)

                poly = poly.reshape(-1, 2)
                cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=4)
                ptColor = (0, 255, 255)
                if verticals is not None:
                    if verticals[i]:
                        ptColor = (255, 0, 0)

                if texts is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

        # Save result image
        cv2.imwrite(res_img_path, img)

