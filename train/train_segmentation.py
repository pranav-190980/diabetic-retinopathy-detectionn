import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.unet import UNet

# Initialize model
model = UNet().cuda()

# Loss + optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TODO: Replace with actual dataset
train_loader = DataLoader(...)

for epoch in range(10):
    model.train()

    total_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()

        preds = model(imgs)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), "outputs/models/unet.pth")
