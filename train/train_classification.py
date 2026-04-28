import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.classifier import get_model
from utils.dataset import DRDataset

# Load dataset
dataset = DRDataset("train.csv", "data/train")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load model
model = get_model("efficientnet").cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()

    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

# Save model
torch.save(model.state_dict(), "outputs/models/classifier.pth")
