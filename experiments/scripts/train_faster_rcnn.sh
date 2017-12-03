#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
ITER=$4
ECCV=$5
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    ITERS=110000
    ANCHORS="[8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_val"
    TEST_IMDB="coco_2014_val"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery)
    TRAIN_IMDB="grocery_train"
    TEST_IMDB="grocery_test"
    STEPSIZE="[80000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery2)
    TRAIN_IMDB="grocery2_train"
    TEST_IMDB="grocery2_test"
    STEPSIZE="[3000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery3)
    TRAIN_IMDB="grocery3_train"
    TEST_IMDB="grocery3_test"
    STEPSIZE="[10000,20000,30000,40000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery4)
    TRAIN_IMDB="grocery4_train"
    TEST_IMDB="grocery4_test"
    STEPSIZE="[10000,20000,30000,40000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery5)
    TRAIN_IMDB="grocery5_train"
    TEST_IMDB="grocery5_test"
    STEPSIZE="[10000,20000,30000,40000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery6)
    TRAIN_IMDB="grocery6_train"
    TEST_IMDB="grocery6_test"
    STEPSIZE="[10000,20000,30000,40000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery7)
    TRAIN_IMDB="grocery7_train"
    TEST_IMDB="grocery7_test"
    #STEPSIZE="[10000,20000,30000,40000]"
    STEPSIZE="[5000,10000,15000,20000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery_full)
    TRAIN_IMDB="grocery_full_train"
    TEST_IMDB="grocery_full_test"
   # STEPSIZE="[20000,40000]"
    STEPSIZE="[25000,40000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery8)
    TRAIN_IMDB="grocery8_train"
    TEST_IMDB="grocery8_test"
    #STEPSIZE="[10000,20000,30000,40000]"
    STEPSIZE="[15000,30000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery9)
    TRAIN_IMDB="grocery9_train"
    TEST_IMDB="grocery9_test"
    #STEPSIZE="[10000,20000,30000,40000]"
    STEPSIZE="[15000,30000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery10)
    TRAIN_IMDB="grocery10_train"
    TEST_IMDB="grocery10_test"
    #STEPSIZE="[10000,20000,30000,40000]"
    STEPSIZE="[8000,18000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  grocery11)
    TRAIN_IMDB="grocery11_train"
    TEST_IMDB="grocery11_test"
    #STEPSIZE="[10000,20000,30000,40000]"
    STEPSIZE="[10000,20000]"
    ITERS=${ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *) 
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/trainval_net.py \
      --weight output/res101/coco_2014_train+coco_2014_valminusminival/default/res101_faster_rcnn_iter_1190000.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_faster_rcnn.sh $@
