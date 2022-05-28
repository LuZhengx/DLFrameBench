#!/bin/bash

net=(resnet20 resnet56 resnet110)
batch=(128)

DEVICE_ID="0"

while getopts o:d: OPT; do
 case ${OPT} in
  d) DEVICE_ID=${OPTARG}
    ;;
  \?)
    printf "[Usage] baseline_test.sh -d <DEVICE_ID>\n" >&2
    exit 1
 esac
done

export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

for i in ${net[@]}; do
    for b in ${batch[@]}; do
        if [ ! -d experiments/baseline ]; then
            mkdir -p experiments/baseline
        fi
        echo $i-$b...
        python train.py --epochs=160 \
                        --batch-size=$b \
                        --arch=$i \
                        > experiments/baseline/$i-$b-4.log
    done
done

echo done!