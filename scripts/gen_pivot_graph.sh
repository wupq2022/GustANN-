#!/bin/bash

HOME_DIR=$(readlink -f $(dirname $0)/..)
DISKANN_DIR=$HOME_DIR/deps/DiskANN
source $HOME_DIR/scripts/setup.sh

SAMPLES=1000000

NAV_OUTPUT=$PIVOT_GRAPH
mkdir -p $NAV_OUTPUT

$HOME_DIR/build/bin/gen_small_file $SAMPLES $DATA_SIZE $DATASET_FILE $NAV_OUTPUT/data.bin $NAV_OUTPUT/map.txt
$DISKANN_DIR/build/apps/build_memory_index --data_type $DATA_TYPE --dist_fn l2 --index_path_prefix $NAV_OUTPUT/nav_index --data_path $NAV_OUTPUT/data.bin -R 32 -L 50 
python3 $HOME_DIR/scripts/gen_tag.py $NAV_OUTPUT/map.txt $NAV_OUTPUT/nav_index.tags
