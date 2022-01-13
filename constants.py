# Training
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 1000

# Architecture
DROPOUT = 0.2
NUM_CHANNELS = 32
DEPTH = 4

# Graph U-Nets
TOP_K_POOLING = 0.5

# GAT Nets
NUM_HEADS = 8

# PNA Nets
AGGREGATORS = ["mean", "std", "max", "min"]
SCALERS = ["identity", "amplification", "attenuation"]
NUM_TOWERS = 4
PRE_LAYERS = 1
POST_LAYERS = 1
