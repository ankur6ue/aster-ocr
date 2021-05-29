import unittest
import json
import os
import numpy as np
import logging
import OCR as ocr
from utils import loadImage, write_text_detection_results, write_recognition_results, divide_chunks


env_file = 'env_local.list'
if os.path.isfile(env_file):
    with open(env_file) as f:
        for line in f:
            name, var = line.partition("=")[::2]
            os.environ[name] = var.rstrip()  # strip trailing newline


class TestDetectionMethods(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.abspath(os.getcwd())
        with open('det_args.json') as f:
            self.det_args = json.load(f)
        path = os.environ.get("DET_MODEL_PATH")
        self.det_args["trained_model"] = os.path.join(base_dir, path)
        self.det_net = ocr.init_detection_model(self.det_args)

    # a normal size input image with 23 words
    def test_detection_1(self):
        img_name = 'paper-title.jpg'
        img = loadImage(img_name)
        ocr.text_detection_results = ocr.text_detect(img, self.det_net, self.det_args)
        # check that 23 boxes are found in this input image
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 23)

    def test_detection_2(self):
        img_name = 'spotting.jpg'
        img = loadImage(img_name)
        ocr.text_detection_results = ocr.text_detect(img, self.det_net, self.det_args)
        # check that 1 box is found in this input image
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 1)

    def test_detection_3(self):
        # run detection on a images of zeros
        zero_img = np.zeros((50, 50, 3), np.uint8)
        ocr.text_detection_results = ocr.text_detect(zero_img, self.det_net, self.det_args)
        # check that 0 boxes are found in this input image
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 0)

    def test_detection_4(self):
        # run detection on an image consisting of random numbers
        rand_img = np.random.randint(0, 255, (50, 50, 3), np.uint8)
        ocr.text_detection_results = ocr.text_detect(rand_img, self.det_net, self.det_args)
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 0)

    def test_detection_5(self):
        # run detection on an image of size 1, 1, 3
        small_img = np.random.randint(0, 255, (1, 1, 3), np.uint8)
        ocr.text_detection_results = ocr.text_detect(small_img, self.det_net, self.det_args)
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 0)

    def test_detection_6(self):
        # run detection on an image of size 0, 0, 3
        rand_img = np.random.randint(0, 255, (0, 0, 3), np.uint8)
        with self.assertRaises(ValueError):
            ocr.text_detection_results = ocr.text_detect(rand_img, self.det_net, self.det_args)

    def test_detection_7(self):
        # incorrect data type for input image
        bad_datatype_img = np.random.randint(0, 255, (0, 0, 3), np.uint16)
        with self.assertRaises(TypeError):
            ocr.text_detection_results = ocr.text_detect(bad_datatype_img, self.det_net, self.det_args,
                                                         logging.getLogger('test'))


class TestRecognitionMethods(unittest.TestCase):
    def setUp(self):
        base_dir = os.path.abspath(os.getcwd())
        with open('det_args.json') as f:
            self.det_args = json.load(f)
        path = os.environ.get("DET_MODEL_PATH")
        self.det_args["trained_model"] = os.path.join(base_dir, path)
        self.det_net = ocr.init_detection_model(self.det_args)
        with open('rec_args.json') as f:
            self.rec_args = json.load(f)
        path = os.environ.get("REC_MODEL_PATH")
        self.rec_args["trained_model"] = os.path.join(base_dir, path)
        self.rec_net = ocr.init_rec_model(self.rec_args)

    # a normal size input image with 23 words
    def test_recognition_1(self):
        img_name = 'paper-title.jpg'
        img = loadImage(img_name)
        ocr.text_detection_results = ocr.text_detect(img, self.det_net, self.det_args)
        # check that 23 boxes are found in this input image
        self.assertEqual(len(ocr.text_detection_results['bboxes']), 23)
        # get crops from bounding boxes and input image
        bboxes = ocr.text_detection_results['bboxes']
        crops = {}
        for k, v in bboxes.items():
            bbox_coords = v.split(',')
            min_x = (int)(bbox_coords[0])
            min_y = (int)(bbox_coords[1])
            max_x = (int)(bbox_coords[4])
            max_y = (int)(bbox_coords[5])
            crops[k] = img[min_y:max_y, min_x:max_x, :]
        recognition_results = ocr.recognize(crops, self.rec_args, self.rec_net)
        self.assertEqual(len(recognition_results), 23)
        self.assertEqual(recognition_results['crop0']['result'], 'endtoend')


    def test_recognition_2(self):
        img_name = 'spotting.jpg'
        img = loadImage(img_name)
        recognition_results = ocr.recognize({'crop0': img}, self.rec_args, self.rec_net)
        self.assertEqual(len(recognition_results), 1)
        self.assertEqual(recognition_results['crop0']['result'], 'spotting')

    def test_recognition_3(self):
        # run recognition on a images of zeros
        zero_img = np.zeros((50, 50, 3), np.uint8)
        recognition_results = ocr.recognize({'crop0': zero_img}, self.rec_args, self.rec_net)
        # Even for a image full of zeros, the ocr.recognizer returns a random string..
        self.assertEqual(len(recognition_results), 1)

    def test_recognition_4(self):
        # run recognition on an image consisting of random numbers
        rand_img = np.random.randint(0, 255, (50, 50, 3), np.uint8)
        recognition_results = ocr.recognize({'crop0': rand_img}, self.rec_args, self.rec_net)
        self.assertEqual(len(recognition_results), 1)

    def test_recognition_5(self):
        # run recognition on an image of size 1, 1, 3
        small_img = np.random.randint(0, 255, (1, 1, 3), np.uint8)
        # check the system doesn't raise an exception
        recognition_results = ocr.recognize({'crop0': small_img}, self.rec_args, self.rec_net)
        self.assertEqual(len(recognition_results), 1)

    def test_recognition_6(self):
        # run recognition on an image of size 0, 0, 3
        rand_img = np.random.randint(0, 255, (0, 0, 3), np.uint8)
        with self.assertRaises(ValueError):
            recognition_results = ocr.recognize({'crop0': rand_img}, self.rec_args, self.rec_net)

    def test_recognition_7(self):
        # run recognition on image with wrong data type
        rand_img = np.random.randint(0, 255, (1, 1, 3), np.uint16)
        with self.assertRaises(TypeError):
            recognition_results = ocr.recognize({'crop0': rand_img}, self.rec_args, self.rec_net)
            

if __name__ == '__main__':
    suite = unittest.TestSuite()
    for i in range(1, 8):
        test_name = "test_recognition_" + str(i)
        suite.addTest(TestRecognitionMethods(test_name))
    # Add all tests
    for i in range(1, 8):
        test_name = "test_detection_" + str(i)
        suite.addTest(TestDetectionMethods(test_name))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    # unittest.main()