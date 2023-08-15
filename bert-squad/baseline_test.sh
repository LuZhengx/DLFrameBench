#!/bin/bash

batch=(32)
number=(1 2 3 4 5)

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

for n in ${number[@]}; do
    if [ ! -d experiments/baseline ]; then
        mkdir -p experiments/baseline
    fi
    echo $n...
    python train.py --bert_model=../pretrained/uncased_L-12_H-768_A-12/ \
                    --output_dir=./ \
                    --do_lower_case \
                    --num_train_epochs=2 \
                    --disable-progress-bar \
                    > experiments/baseline/uncased_L-12_H-768_A-12-$n.log 2>&1
done

echo done!