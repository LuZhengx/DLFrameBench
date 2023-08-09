config = {}

# ============== Model Layers ==============
config['ln-epsilon'] = 1e-12
config['ln-gamma-init'] = 'Ones'
config['ln-beta-init'] = 'Zeros'

config['dense-use-bias'] = True
config['dense-weight-init'] = 'XavierUniform'
config['dense-bias-init'] = 'Zeros'

# =========== Dataset Infomation ===========
config['train-file'] = '/data/dataset/squadv1.1/train-v1.1.json'
config['predict-file'] = '/data/dataset/squadv1.1/dev-v1.1.json'
config['eval_script'] = 'evaluate-v1.1.py'
config['dataset-num-workers'] = 8

# ======== Training Hyperparameters ========
config['optimizer-type'] = 'AdamW'
config['weight-decay'] = 0.01

config['lr-base'] = 5e-5
config['lr-batch-denom'] = 32
config['lr-warmup-proportion'] = 0.1
config['train-batch-size'] = 32
config['predict-batch-size'] = 32
