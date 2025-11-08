# Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import ipywidgets
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Only print versions once when module is first imported
if __name__ == '__main__':
    print("Python version", sys.version)
    print("PyTorch version", torch.__version__)
    print("Torchvision version", torchvision.__version__)
    print("Numpy version", np.__version__)
    print("Scipy version", sp.__version__)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    print(f'Using device: {device}')

# ==================== CONFIGURATION ====================
class Config:
    # Paths (modify these to point to your dataset)
    DATA_DIR = 'images'
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15  # Remaining 15%
    
    # Model parameters
    MODEL_NAME = 'efficientnet_b3'  # Modern, efficient architecture
    PRETRAINED = True
    IMG_SIZE = 300
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    
    # Training parameters
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    
    # Scheduler parameters
    SCHEDULER = 'cosine'  # 'cosine' or 'plateau'
    MIN_LR = 1e-6
    
    # Early stopping
    PATIENCE = 7
    
    # Mixed precision training (for faster training on compatible devices)
    USE_AMP = True
    
    # Random seed
    SEED = 42

config = Config()

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.SEED)

# ==================== DATA AUGMENTATION ====================
# Training transforms with aggressive augmentation
train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== DATASET LOADING ====================
def split_dataset(dataset, train_ratio, val_ratio):
    """Split dataset into train, validation, and test sets"""
    from torch.utils.data import random_split
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.SEED)
    )
    
    return train_dataset, val_dataset, test_dataset

def load_datasets():
    print("Loading datasets...")
    print(f"Dataset path: {config.DATA_DIR}")
    
    # Load full dataset with training transforms
    full_dataset = ImageFolder(root=config.DATA_DIR, transform=None)
    
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset, 
        config.TRAIN_RATIO, 
        config.VAL_RATIO
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)} ({config.TRAIN_RATIO*100:.0f}%)")
    print(f"  Validation samples: {len(val_dataset)} ({config.VAL_RATIO*100:.0f}%)")
    print(f"  Test samples: {len(test_dataset)} ({config.TEST_RATIO*100:.0f}%)")
    
    # Apply transforms to each split
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = val_transforms
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type in ['cuda', 'mps'] else False
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names

# ==================== MODEL ARCHITECTURE ====================
class WasteClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(WasteClassifier, self).__init__()
        
        # Load pretrained model from timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classifier head
        )
        
        # Get number of features from backbone
        num_features = self.backbone.num_features
        
        # Custom classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ==================== TRAINING UTILITIES ====================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop

class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc', linewidth=2)
        ax2.plot(self.val_accs, label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ==================== TRAINING LOOP ====================
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if config.USE_AMP and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# ==================== MAIN TRAINING FUNCTION ====================
def train_model():
    # Load data
    train_loader, val_loader, test_loader, num_classes, class_names = load_datasets()
    
    # Initialize model
    print(f"\nInitializing {config.MODEL_NAME} model...")
    model = WasteClassifier(config.MODEL_NAME, num_classes, config.PRETRAINED)
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    if config.SCHEDULER == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.NUM_EPOCHS, 
            eta_min=config.MIN_LR
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5,
            min_lr=config.MIN_LR
        )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and device.type == 'cuda' else None
    
    # Early stopping and metrics
    early_stopping = EarlyStopping(patience=config.PATIENCE, mode='max')
    metrics = MetricsTracker()
    
    best_val_acc = 0.0
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update metrics
        metrics.update(train_loss, train_acc, val_loss, val_acc)
        
        # Learning rate scheduling
        if config.SCHEDULER == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, 'best_waste_classifier.pth')
            print(f'✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
        
        # Early stopping
        if early_stopping(val_acc):
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
        
        print()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*50 + "\n")
    
    # Plot training history
    metrics.plot()
    
    # Test evaluation if test set exists
    if test_loader:
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load('best_waste_classifier.pth')['model_state_dict'])
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
    
    return model, class_names

# ==================== INFERENCE UTILITIES ====================
def load_trained_model(model_path='best_waste_classifier.pth'):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    model = WasteClassifier(config.MODEL_NAME, num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Classes: {class_names}")
    return model, class_names

def predict_image(model, image_path, class_names, show_plot=True):
    """Predict class for a single image"""
    from PIL import Image
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = val_transforms(img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence_score:.2f}%')
        
        plt.subplot(1, 2, 2)
        probs = probabilities.cpu().numpy()[0] * 100
        plt.barh(class_names, probs)
        plt.xlabel('Confidence (%)')
        plt.title('Class Probabilities')
        plt.tight_layout()
        plt.show()
    
    return predicted_class, confidence_score

# ==================== RUN TRAINING ====================
if __name__ == '__main__':
    # Train the model
    model, class_names = train_model()
    
    # Example inference (uncomment to use)
    # model, class_names = load_trained_model()
    # predict_image(model, 'path/to/test/image.jpg', class_names)
