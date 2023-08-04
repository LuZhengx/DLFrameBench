# Fine-tuning BERT on SQuAD dataset

## pretrained model pre-processing

1. Download tensorflow model (.zip) from official website.

2. Unzip the model into `./pretrained` folder.

3. Running `convert_tf_to_numpy.py` in the environment with TensorFlow 2 to convert the weights into numpy file.

``` shell
python convert_tf_to_numpy.py --tf_path=./pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt
```

## start fine tuning

Run the script `train.py` in the framework folder with the following args: `bert_model`, `output_dir`, `train_file`, and `predict_file`.

``` shell
python train.py --bert_model=../pretrained/uncased_L-12_H-768_A-12/ --output_dir=./ --predict_file=/data/dataset/squadv1.1/dev-v1.1.json --train_file=/data/dataset/squadv1.1/train-v1.1.json --do_lower_case
```

To load a exist fine tuned model, use the following command:

```shell
python train.py --bert_model=../pretrained/uncased_L-12_H-768_A-12/ --output_dir=./ --predict_file=/data/dataset/squadv1.1/dev-v1.1.json --train_file=/data/dataset/squadv1.1/train-v1.1.json --do_lower_case --init_checkpoint=/data/dck/bert/bert_base_qa.pt --config_file=/data/dck/bert/nvidia_pytorch/bert_configs/base.json
```