import logging
from src.configs import Config
from src.trainer import Trainer
import os

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Create necessary directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Print dataset information
    logging.info(f"Data directory: {Config.DATA_DIR}")

    if not os.path.exists(Config.DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {Config.DATA_DIR}")
    
    # Initialize trainer
    trainer = Trainer(Config)
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    main()