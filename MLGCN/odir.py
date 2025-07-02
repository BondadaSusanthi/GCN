import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import pickle
from gcn_model import gcn_resnet101
from engine import Engine


class ODIRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.labels = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.data.iloc[idx]['image']}"
        image = Image.open(img_path).convert('RGB')

        label = torch.FloatTensor(self.data.iloc[idx][self.labels].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Paths
    train_csv = 'data/odir_train.csv'
    val_csv = 'data/odir_val.csv'
    img_dir = 'data/images'
    adj_file = 'data/adj.pkl'
    embed_file = 'data/embeddings.pkl'

    # Hyperparameters
    batch_size = 8
    epochs = 10
    lr = 0.001

    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = ODIRDataset(train_csv, img_dir, transform)
    val_dataset = ODIRDataset(val_csv, img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(train_dataset.labels)

    # Model
    model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file=adj_file)
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # Load label embeddings
    with open(embed_file, 'rb') as f:
        label_embed = pickle.load(f)
    label_embed = torch.from_numpy(label_embed).float()

    # Engine
    engine = Engine()
    engine.label_embed = label_embed

    for epoch in range(epochs):
        engine.train(train_loader, model, criterion, optimizer, epoch)
        engine.validate(val_loader, model, criterion)


if __name__ == "__main__":
    main()
