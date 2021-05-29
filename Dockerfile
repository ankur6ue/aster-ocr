# FROM prefecthq/prefect:0.14.5-python3.7 as base_image
FROM python:3.8 as base_image

# Install pip
RUN python -m pip install --upgrade pip

ENV PREFECT__USER_CONFIG_PATH='/opt/prefect/config.toml'
ENV BASE_DIR=/opt/ocr
# RUN pip show prefect || pip install git+https://github.com/PrefectHQ/prefect.git@0.14.5#egg=prefect[all_orchestration_extras]
RUN pip install wheel
RUN pip install torch torchvision
# needed to get rid of ImportError: libGL.so.1:
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# needed to get rid of ImportError: libtk8.6.so
RUN apt-get install tk -y
RUN apt-get install nano
RUN mkdir -p $BASE_DIR
COPY requirements_docker.txt $BASE_DIR/requirements.txt
WORKDIR $BASE_DIR
RUN pip install -r requirements.txt
# install OCR python package
COPY setup.py $BASE_DIR/
COPY README.md $BASE_DIR/
RUN python setup.py bdist_wheel
# install the OCR package from dist
RUN pip install dist/OCR*.whl
# copy the models into the base image
COPY models/detection-CRAFT/craft_mlt_25k.pth $BASE_DIR/models/detection-CRAFT/craft_mlt_25k.pth
COPY models/recognition-ASTER/demo.pth.tar $BASE_DIR/models/recognition-ASTER/demo.pth.tar


FROM base_image as test
WORKDIR $BASE_DIR
COPY rec_args.json $BASE_DIR/
COPY det_args.json $BASE_DIR/
COPY paper-title.jpg $BASE_DIR/
COPY env_local.list $BASE_DIR/
COPY *.py $BASE_DIR/


# to build just the base_image:
# docker build --target base_image -t myocr .
# omit --target to build test image


