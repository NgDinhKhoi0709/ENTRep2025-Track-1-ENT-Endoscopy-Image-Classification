import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import surgvlp
from mmengine.config import Config
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

class EndoscopeClassifier(nn.Module):
    def __init__(self, peskaVLP_model, num_classes=7, freeze_encoder=True):
        super().__init__()
        # Store the whole PeskaVLP model to use its image encoding functionality
        self.peskaVLP = peskaVLP_model
        
        # Freeze the PeskaVLP model if specified
        if freeze_encoder:
            for param in self.peskaVLP.parameters():
                param.requires_grad = False
                
        # Get the embedding dimension from the model
        embedding_dim = 768
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Use the entire PeskaVLP model to get image features
        with torch.no_grad():
            features = self.peskaVLP(x, None, mode='video')['img_emb']

            # Ensure we have the right shape
            if isinstance(features, tuple):
                features = features[0]
            if features.dim() > 2:
                features = features[:, 0]  # Get the [CLS] token features
                
        # Pass through classification head
        return self.classifier(features)

class ENTJsonDataset(Dataset):
    """Dataset backed by the Train json annotation list."""

    LABELS = ['nose-right', 'nose-left', 'ear-right', 'ear-left', 'vc-open', 'vc-closed', 'throat']

    def __init__(self, items, img_dir, transform=None):
        self.items = items  # list of dicts from json
        self.img_dir = img_dir
        self.transform = transform
        self.label_to_idx = {c: i for i, c in enumerate(self.LABELS)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_name = item['Path']
        label_str = item['Classification']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.label_to_idx[label_str], dtype=torch.long)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # accumulate for balanced accuracy
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(predicted.cpu().tolist())
        
        progress_bar.set_postfix({
            'loss': total_loss/len(dataloader),
            'acc': 100.*correct/total
        })
        
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    return total_loss/len(dataloader), 100.*correct/total, bal_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # accumulate for balanced accuracy
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())
            
    bal_acc = balanced_accuracy_score(all_targets, all_preds) * 100.0
    return total_loss/len(dataloader), 100.*correct/total, bal_acc

# ------------------------------------------------------------------
# Loss: FocalLoss implementation
# ------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Args:
        alpha (float): balancing factor for positive/negative examples.
        gamma (float): focusing parameter that reduces the relative loss
                       for well-classified examples.
        reduction (str): "none", "mean", or "sum".
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # Compute standard cross-entropy loss (per sample)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load PeskaVLP model
    configs = Config.fromfile('./SurgVLP/tests/config_peskavlp.py')['config']
    peskaVLP_model, preprocess = surgvlp.load(configs.model_config, device=device, pretrain='./SurgVLP/weights/PeskaVLP.pth')
    
    # Create our classifier
    model = EndoscopeClassifier(peskaVLP_model, num_classes=7, freeze_encoder=True)
    model = model.to(device)

    # Dataset and DataLoader
    train_transform = preprocess
    val_transform   = preprocess
    
    # ------------------------------------------------------------------
    # Prepare ENT dataset – use ALL data for training as specified by cls.json
    # ------------------------------------------------------------------
    json_path = Path('./dataset/augmented_merge_original/cls_train.json')
    img_dir   = Path('./dataset/augmented_merge_original/images')

    if not json_path.exists():
        raise FileNotFoundError(f'Missing label file: {json_path}')

    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    records = [{'Path': fn, 'Classification': lbl} for fn, lbl in mapping.items()]

    train_dataset = ENTJsonDataset(records, img_dir, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path = ''
    start_epoch = 0
    best_val_acc = 0
    if Path(ckpt_path).is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print('Optimizer state mismatch, continuing with fresh optimizer')
        best_val_acc = ckpt.get('val_accuracy', 0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from checkpoint {ckpt_path} (epoch {start_epoch}, best val acc {best_val_acc:.2f}%)")

    # Training loop – continue up to total_epochs
    total_epochs = 10  # change as needed
    
    best_bal_acc = 0.0
    best_acc     = 0.0
    patience      = 3   # epochs to wait for improvement before stopping
    epochs_no_improve = 0

    for epoch in range(start_epoch, total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")
        
        # Training phase (entire dataset)
        train_loss, train_acc, train_bal_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Training - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, BalAcc: {train_bal_acc:.2f}%")
        
        # Update learning rate based on training loss
        scheduler.step(train_loss)
        
        # Save the best model based on balanced accuracy calculated on the whole training set
        if train_bal_acc > best_bal_acc:
            best_bal_acc = train_bal_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bal_accuracy': best_bal_acc,
            }, './weights/best_balacc.pth')
            print(f"Saved new best model with balanced accuracy: {train_bal_acc:.2f}%")
            epochs_no_improve = 0  # reset counter when improvement occurs

        # Save best model based on overall accuracy
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
            }, './weights/best_acc.pth')
            print(f"Saved new best model with accuracy: {train_acc:.2f}%")

        # Increment early stopping counter if no improvement in bal acc
        if train_bal_acc <= best_bal_acc:
            epochs_no_improve += 1

        # Check early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # ------------------------------------------------------------------
    # Save last model after training completes
    # ------------------------------------------------------------------
    torch.save({
        'epoch': total_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'bal_accuracy': train_bal_acc,
    }, './weights/last.pth')
    print("Saved last model after final epoch.")

if __name__ == '__main__':
    main()
