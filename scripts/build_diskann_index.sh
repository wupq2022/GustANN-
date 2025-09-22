#!/bin/bash

HOME_DIR=$(readlink -f $(dirname $0)/..)
DISKANN_DIR=$HOME_DIR/deps/DiskANN
source $HOME_DIR/scripts/setup.sh

$DISKANN_DIR/build/apps/build_disk_index --data_type $DATA_TYPE --dist_fn l2 --index_path_prefix $INDEX_PREFIX --data_path $DATASET_FILE -B $1 -M $2 -R 128 -L 200
