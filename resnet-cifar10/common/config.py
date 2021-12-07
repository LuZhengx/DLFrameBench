config = {}

# ============== Model Layers ==============
config['conv2d-use-bias'] = False
config['conv2d-weight-init'] = 'XavierUniform'
config['conv2d-pytorch-style'] = False

config['bn-momentum'] = 0.9
config['bn-epsilon'] = 1e-5
config['bn-gamma-init'] = 'Ones'
config['bn-beta-init'] = 'Zeros'

config['dense-use-bias'] = True
config['dense-weight-init'] = 'XavierUniform'
config['dense-bias-init'] = 'Zeros'

config['downsample-shortcut'] = 'convolution'

# =========== Dataset Infomation ===========
config['dataset-train-size'] = 50000
config['dataset-val-size'] = 10000
config['dataset-img-size'] = [3, 32, 32] #CHW
config['dataset-num-workers'] = 8

# =============== Preprocess ===============
config['dataset-img-pad'] = 4
config['dataset-img-means'] = [0.485, 0.456, 0.406]
config['dataset-img-stds'] = [0.229, 0.224, 0.225]

# ======== Training Hyperparameters ========
config['optimizer-type'] = 'SGD'
config['sgd-momentum'] = 0.9
config['sgd-nesterov'] = False
config['sgd-weight-decay'] = 2e-4

config['lr-base'] = 0.1
config['lr-decay-boundaries'] = [80, 120]
config['lr-decay-rate'] = 0.1
config['lr-batch-denom'] = 128
config['lr-warmup-rate'] = 0.1
config['lr-warmup-epochs'] = 1