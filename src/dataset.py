import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

class WaterBirdDataset(Dataset):
    """Dataset class for WaterBird classification with domain adaptation"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Optional transforms to be applied on images
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        # Verify the split directory exists
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Split directory '{self.split_dir}' not found in root directory '{root_dir}'")
        
        self.image_paths = []
        self.labels = []

        # Load images and labels
        for label in ['0', '1']:  # Assuming labels are "0" (landbird) and "1" (waterbird)
            label_dir = os.path.join(self.split_dir, label)
            if not os.path.exists(label_dir):
                logging.warning(f"Label directory '{label_dir}' not found for split '{split}'")
                continue
            
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(int(label))  # Convert label to int

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid images found in '{self.split_dir}' for split '{split}'")
        
        logging.info(f"Loaded {len(self.image_paths)} images for split '{split}'")

        # Set up transforms
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'bird_label': torch.tensor(label, dtype=torch.long),
                'filename': os.path.basename(img_path)
            }
        
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise
