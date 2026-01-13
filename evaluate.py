import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

def run_test_and_ood():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Define transforms (identical to validation)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2) Load test data
    test_dataset = datasets.ImageFolder('final_dataset/test', transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes

    # 3) Load trained model
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 13)
    model.load_state_dict(torch.load('chess_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    ood_count = 0
    
    # Confidence threshold: if the model is < 70% confident, count it as OOD
    CONFIDENCE_THRESHOLD = 0.70 

    print("Evaluating Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Convert logits to probabilities (Softmax)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, 1)

            for i in range(len(preds)):
                if confidences[i] < CONFIDENCE_THRESHOLD:
                    # Detected as Out-of-Distribution
                    ood_count += 1
                
                all_preds.append(preds[i].item())
                all_labels.append(labels[i].item())

    # 4) Report results
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {acc:.4f}")
    print(f"OOD Detections (Low Confidence): {ood_count} images out of {len(all_labels)}")

    # 5) Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

if __name__ == "__main__":
    run_test_and_ood()