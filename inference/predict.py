import torch
from models.classifier import get_model
from utils.preprocessing import preprocess_image

# Load model
model = get_model("efficientnet")
model.load_state_dict(torch.load("outputs/models/classifier.pth"))
model.eval()

# Load image
img = preprocess_image("test.jpg")
img = torch.tensor(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, dim=1)

print("Predicted Class:", pred.item())
