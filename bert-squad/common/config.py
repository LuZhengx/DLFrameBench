config = {}

# ============== Model Layers ==============
config['bn-momentum'] = 0.9
config['bn-epsilon'] = 1e-5
config['bn-gamma-init'] = 'Ones'
config['bn-beta-init'] = 'Zeros'

config['dense-use-bias'] = True
config['dense-weight-init'] = 'XavierUniform'
config['dense-bias-init'] = 'Zeros'

# =========== Dataset Infomation ===========
config['dataset-num-workers'] = 8

# ======== Training Hyperparameters ========
config['optimizer-type'] = 'AdamW'
config['weight-decay'] = 0.01

config['lr-base'] = 5e-5
config['lr-batch-denom'] = 32
config['lr-warmup-proportion'] = 0.1