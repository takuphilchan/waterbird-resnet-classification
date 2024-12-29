import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'waterbirds_dataset')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Model
    MODEL_NAME = 'resnet18'
    NUM_CLASSES = 2
    PRETRAINED = True
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    IMG_SIZE = 224
    NUM_WORKERS = 4
    
    # Logging
    LOG_INTERVAL = 10
    VAL_INTERVAL = 1
