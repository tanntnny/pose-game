
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from utils.dataset import PoseDataset
from utils.classifier import PoseClassifier
from sklearn.model_selection import train_test_split

# Paths
DATA_CSV = os.path.join(os.path.dirname(__file__), 'data/data.csv')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models/pose_classifier.pth')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3


# Dataset and Train/Test Split
full_dataset = PoseDataset(DATA_CSV)
indices = list(range(len(full_dataset)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
input_size = full_dataset.X.shape[1] if len(full_dataset.X.shape) > 1 else 1
num_classes = full_dataset.Y.shape[1] if len(full_dataset.Y.shape) > 1 else 1
model = PoseClassifier(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)
            loss = criterion(outputs, y)
            test_loss += loss.item() * X.size(0)
    test_loss = test_loss / len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
    # Save best model
    if test_loss < best_loss:
        best_loss = test_loss
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")
