#!/bin/bash

net=(12)
batch=(1 2 4 8 16 32)
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
                    python train.py --bert_model=../pretrained/uncased_L-12_H-768_A-12/ \
                                    --output_dir=./ \
                                    --do_lower_case \
                                    --num_train_epochs=6 \
                                    --disable-progress-bar \
                                    --train-batch-size=$b \
                                    --is-prof \
                                    > experiments/$i-$b/${OUTFILE_NAME}.log 2>&1
    done
done

echo done!