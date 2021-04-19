import os
import json
import torch
from utils import loadImage, write_text_detection_results, write_recognition_results
import OCR as ocr

torch.set_num_threads(1)

# read input image

env_file = 'env_local.list'
if os.path.isfile(env_file):
    with open(env_file) as f:
        for line in f:
            name, var = line.partition("=")[::2]
            os.environ[name] = var.rstrip()  # strip trailing newline


if __name__ == '__main__':
    img_name = 'paper-title.jpg'
    base_dir = os.path.abspath(os.getcwd())
    base_file_name, ext = os.path.splitext(img_name)
    img = loadImage(img_name)
    with open('det_args.json') as f:
        det_args = json.load(f)
    det_path = os.environ.get("DET_MODEL_PATH")
    det_args["trained_model"] = os.path.join(base_dir, det_path)
    det_net = ocr.init_detection_model(det_args)

    with open('rec_args.json') as f:
        rec_args = json.load(f)
    rec_path = os.environ.get("REC_MODEL_PATH")
    rec_args["trained_model"] = os.path.join(base_dir, rec_path)
    rec_model = ocr.init_rec_model(rec_args)

    text_detection_results = ocr.text_detect(img, det_net, det_args)
    write_text_detection_results(base_file_name, img, text_detection_results)
    # get crops from bounding boxes and input image
    bboxes = text_detection_results['bboxes']
    crops = {}
    for k, v in bboxes.items():
        bbox_coords = v.split(',')
        min_x = (int)(bbox_coords[0])
        min_y = (int)(bbox_coords[1])
        max_x = (int)(bbox_coords[4])
        max_y = (int)(bbox_coords[5])
        crops[k] = img[min_y:max_y, min_x:max_x, :]
    recognition_results = ocr.recognize(crops, rec_args, rec_model)
    write_recognition_results(base_file_name, recognition_results)


