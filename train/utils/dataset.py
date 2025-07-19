from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = []
        self.Y = []
        num_classes = df['label'].nunique()

        for _, row in df.iterrows():
            filename = row['filename']
            label = int(row['label'])
            data = np.load(filename)
            self.X.extend(data.tolist())
            # Create one-hot encoded labels for each sample in data
            for _ in range(len(data)):
                one_hot = [0] * num_classes
                one_hot[label] = 1
                self.Y.append(one_hot)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
