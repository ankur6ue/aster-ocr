#!/bin/sh
docker build -t myocr .
SSH_KEY_LOCATION=~/.ssh/home_ubuntu_laptop/id_rsa
WORKER_IPS="../k8s/workers.txt"
REMOTE_DIR=~/dev/apps/ML/OCR
REPOSRC=https://ankur6ue:githubPword123@github.com/ankur6ue/aster-ocr.git
LOCALREPO=aster-ocr
while IFS= read -r line; do
  echo "$line"
  worker_ip=$line
  echo Apollo11 | ssh -tt ankur@$worker_ip -i $SSH_KEY_LOCATION 'mkdir -p' ${REMOTE_DIR} '&& cd' ${REMOTE_DIR} \
    '&& git clone' ${REPOSRC} ${LOCALREPO} '2> /dev/null || (cd' ${LOCALREPO} '; git reset --hard origin/main)'
  # copy model files
  rsync -e "ssh -i $SSH_KEY_LOCATION" -a --relative ./models ankur@$worker_ip:${REMOTE_DIR}/aster-ocr


done <"$WORKER_IPS"
