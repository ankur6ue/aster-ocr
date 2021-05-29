#!/bin/sh
docker run \
--shm-size 8G \
--network="host" \
-i --cpus="5" myocr python ocr_app.py