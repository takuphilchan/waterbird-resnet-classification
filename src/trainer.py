import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.dataset import WaterBirdDataset
from src.model import WaterBirdModel
from src.utils import setup_logging, save_checkpoint

# Free up unused CUDA memory
torch.cuda.empty_cache()

class Trainer:
    """Trainer class for WaterBird classification"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(os.path.join(config.LOG_DIR, 'training.log'))
        
        # Setup model
        self.model = WaterBirdModel(
            num_classes=config.NUM_CLASSES,
            pretrained=config.PRETRAINED
        ).to(config.DEVICE)
        
        # Setup optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Setup data
        self.setup_data()
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def setup_data(self):
        """Initialize datasets and dataloaders"""
        # Create datasets
        self.train_dataset = WaterBirdDataset(
            root_dir=self.config.DATA_DIR,
            split='train'
        )
        
        self.val_dataset = WaterBirdDataset(
            root_dir=self.config.DATA_DIR,
            split='val'
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True
        )
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
            
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training', ncols=100, dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')):
            # Get data
            images = batch['image'].to(self.config.DEVICE)
            bird_labels = batch['bird_label'].to(self.config.DEVICE)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, bird_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += bird_labels.size(0)
            correct += predicted.eq(bird_labels).sum().item()
            
            # Inside your `train_epoch` method, change the logging to avoid interference:
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                tqdm.write(f'Train Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in enumerate(tqdm(self.val_loader, desc='Validation', ncols=100, dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]')):
                images = batch['image'].to(self.config.DEVICE)
                bird_labels = batch['bird_label'].to(self.config.DEVICE)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                # Overall accuracy
                total += bird_labels.size(0)
                correct += predicted.eq(bird_labels).sum().item()
        
        # Calculate overall accuracy
        overall_acc = 100. * correct / total
        return overall_acc
    
    def train(self):
        """Main training loop"""
        self.logger.info(f'Starting training on device: {self.config.DEVICE}')
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.logger.info(f'\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}')
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            if (epoch + 1) % self.config.VAL_INTERVAL == 0:
                val_acc = self.validate()
                self.val_accuracies.append(val_acc)
                
                self.logger.info(f'Validation Accuracy: {val_acc:.2f}%')
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        val_acc,
                        os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
                    )
            
            # Save last model
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_acc,
                os.path.join(self.config.CHECKPOINT_DIR, 'last_model.pth')
            )
