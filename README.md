# Example Package

This package implements basic OCR functionality. It consists of a [text detection](https://ieeexplore.ieee.org/abstract/document/8395027) module and a [text recognition](https://ieeexplore.ieee.org/abstract/document/8395027) module. 

## To build package:
cd to package top level directory and:
python3 setup.py sdist bdist_wheel

## To create conda env:
conda env create -f environment.yml --name ocr_test_env
conda activate ocr_test_env

## To run sample application that runs text detection and recognition on a sample image (paper-title.jpg)
* First download pre-trained detection and recognition models from [here](https://drive.google.com/drive/folders/13V9txMnTGL0Qw2_WhAGNOxSqIfP68595?usp=sharing) and extract to the 
models directory. After extraction, your models directory should look like:
    * /models/detection-CRAFT/craft_mlt_25k.pth
    * /models/recognition-ASTER/demo.pth.tar
    
   If you download the models somewhere else, edit the paths in env_local.list accordingly
   
* Run ```python ocr_app.py```
* To run tests: ```python tests.py```. The tests.py file also provides examples of how to use this package.

## To create requirements.txt
cd to package top level directory and use:
pip3 freeze > requirements.txt






