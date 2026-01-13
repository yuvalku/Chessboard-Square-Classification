import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Define transforms (augmentation for training, simple for validation)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 2) Load data (make sure paths match folders created by split-folders)
    image_datasets = {
        'train': datasets.ImageFolder('final_dataset/train', transform=data_transforms['train']),
        'val': datasets.ImageFolder('final_dataset/val', transform=data_transforms['val'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
    }
    
    # 3) Compute class weights to reduce bias towards empty squares
    class_counts = np.array([len(os.listdir(os.path.join('final_dataset/train', c))) for c in image_datasets['train'].classes])
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    weights_tensor = torch.FloatTensor(weights).to(device)

    # 4) Define the model (ResNet18)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 13) 
    model = model.to(device)

    # 5) Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Helper variables for plots
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 6) Training loop
    num_epochs = 12  # Change as needed
    print("Starting Training...")

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            # Store values for plotting
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'Epoch {epoch+1}/{num_epochs} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save model
    torch.save(model.state_dict(), 'chess_model.pth')
    print("Training Complete. Model saved as chess_model.pth")

    # 7) Create plots (visualization)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='blue')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')  # Save the plot image for the report
    plt.show()

if __name__ == "__main__":
    train_model()