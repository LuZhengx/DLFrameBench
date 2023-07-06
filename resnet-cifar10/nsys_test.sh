#!/bin/bash

net=(resnet20 resnet56 resnet110)
batch=(16 32 64 128 256 512)
# net=(resnet20)
# batch=(128 256)

DEVICE_ID="0"
OUTFILE_NAME="gpu_metric"

while getopts o:d: OPT; do
 case ${OPT} in
  d) DEVICE_ID=${OPTARG}
    ;;
  o) OUTFILE_NAME=${OPTARG}
    ;;
  \?)
    printf "[Usage] nsys_test.sh -d <DEVICE_ID> -o <OUTPUT_FILE_NAME>\n" >&2
    exit 1
 esac
done

export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

for i in ${net[@]}; do
  for b in ${batch[@]}; do
    if [ ! -d experiments/$i-$b ]; then
      mkdir -p experiments/$i-$b
    fi
    echo $i-$b...
    nsys profile -t nvtx \
                 -b none \
                 -o experiments/$i-$b/${OUTFILE_NAME} \
                 -f true \
                 -w true \
                 --sample=none \
                 --cpuctxsw=none \
                 --cudabacktrace=none \
                 --gpu-metrics-device=${DEVICE_ID} \
                 --gpu-metrics-set=ga10x-nvlink \
                 python train.py --epochs=6 \
                                 --batch-size=$b \
                                 --arch=$i \
                                 --is-prof \
                                 > experiments/$i-$b/${OUTFILE_NAME}.log 2>&1
  done
done

echo done!