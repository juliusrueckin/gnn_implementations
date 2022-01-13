# Training
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 10

# Architecture
DROPOUT = 0.2
NUM_CHANNELS = 16
DEPTH = 2

# Graph U-Nets
TOP_K_POOLING = 0.5

# GAT Nets
NUM_HEADS = 8

# PNA Nets
AGGREGATORS = ["mean", "std", "sum", "max", "min"]
SCALERS = ["identity", "amplification", "attenuation"]
NUM_TOWERS = 8
PRE_LAYERS = 1
POST_LAYERS = 1
