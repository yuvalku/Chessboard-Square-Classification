"""Evaluate a trained square-classification model on `final_dataset/test`.

This script:
- Loads a ResNet18 checkpoint/state_dict.
- Runs inference on the test split.
- Applies simple OOD rejection rules.
- Saves a confusion matrix plot under `results/`.
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

CONF_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 0.6

def is_tensor_visually_valid(tensor):
    """
    Simple visual heuristic for rejecting low-information inputs.

    Uses standard deviation of a grayscale approximation of the normalized tensor.
    Very low values often indicate an almost-uniform crop.
    """
    gray_approx = tensor.mean(dim=0) 
    std_dev = torch.std(gray_approx).item()
    return std_dev >= 0.05

def load_evaluation_model(model_path, device):
    """
    Load a ResNet18 model from either a full checkpoint dict or a plain state_dict.
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both full checkpoints and state-only files.
    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model = models.resnet18()
    # Determine output dimension from the saved weights.
    num_classes = state_dict["fc.weight"].shape[0] if "fc.weight" in state_dict else 13
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def run_test_and_ood(model_filename='chess_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Keep transforms consistent with training/validation.
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = 'final_dataset/test'
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found.")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes

    try:
        model = load_evaluation_model(model_filename, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    all_preds = []
    all_labels = []
    
    ood_visual_count = 0
    ood_confidence_count = 0
    ood_entropy_count = 0

    print(f"Evaluating Test Set ({len(test_dataset)} images)...")
    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            
            # Batch inference
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            for i in range(inputs.size(0)):
                img_tensor = inputs[i]
                label = labels[i].item()
                conf = confs[i].item()
                pred = preds[i].item()
                
                # Entropy per sample
                prob_sample = probs[i]
                entropy = -torch.sum(prob_sample * torch.log(prob_sample + 1e-9)).item()

                # OOD rule 1: visual heuristic
                if not is_tensor_visually_valid(img_tensor):
                    ood_visual_count += 1
                    all_preds.append(-1)
                # OOD rule 2: confidence threshold
                elif conf < CONF_THRESHOLD:
                    ood_confidence_count += 1
                    all_preds.append(-1)
                # OOD rule 3: entropy threshold
                elif entropy > ENTROPY_THRESHOLD:
                    ood_entropy_count += 1
                    all_preds.append(-1)
                else:
                    all_preds.append(pred)
                
                all_labels.append(label)
            
            pbar.set_postfix({"OOD": ood_visual_count + ood_confidence_count + ood_entropy_count})

    # Summary
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    valid_mask = all_preds_np != -1
    clean_acc = np.mean(all_preds_np[valid_mask] == all_labels_np[valid_mask]) if np.any(valid_mask) else 0.0

    print("\n" + "="*40)
    print(f"Clean Accuracy (Non-OOD): {clean_acc:.4f}")
    print(f"Total OOD Rejections: {ood_visual_count + ood_confidence_count + ood_entropy_count}")
    print("="*40)

    # Confusion matrix (adds an extra OOD column)
    cm_classes = class_names + ['OOD']
    cm_preds = [p if p != -1 else len(class_names) for p in all_preds]
    cm = confusion_matrix(all_labels, cm_preds, labels=range(len(cm_classes)))
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cm_classes, yticklabels=class_names)
    
    plt.title('Final Confusion Matrix (RGB Pipeline)')
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted (Last Column is OOD)')
    
    report_path = './results/evaluation_confusion_matrix_final.png'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Point to a trained model file
    run_test_and_ood('chess_model.pth')