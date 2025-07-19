import os
import torch
from torch.utils.data import DataLoader
from utils.dataset import PoseDataset
from utils.classifier import PoseClassifier

# Paths
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data/data.csv')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models/pose_classifier.pth')
BATCH_SIZE = 32

# Dataset
full_dataset = PoseDataset(DATA_CSV)
# Use all data for testing (or modify as needed)
test_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
input_size = full_dataset.X.shape[1] if len(full_dataset.X.shape) > 1 else 1
num_classes = full_dataset.Y.shape[1] if len(full_dataset.Y.shape) > 1 else 1
model = PoseClassifier(input_size, num_classes)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

accuracy = correct / total if total > 0 else 0
print(f"Test Accuracy: {accuracy * 100:.2f}%")
