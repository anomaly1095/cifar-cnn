import sys
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
from model import CIFAR10ConvNet10  # your ConvNet10

# -----------------------------
# Check arguments
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python predict.py <path_to_model.pth> <path_to_image>")
    sys.exit(1)

model_path = Path(sys.argv[1])
image_path = Path(sys.argv[2])

if not model_path.exists():
    print(f"Model file not found: {model_path}")
    sys.exit(1)
if not image_path.exists():
    print(f"Image file not found: {image_path}")
    sys.exit(1)

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
model = CIFAR10ConvNet10(num_classes=10).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR10 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -----------------------------
# Load image
# -----------------------------
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)  # add batch dim

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    outputs = model(image)
    pred_class = outputs.argmax(dim=1).item()

# -----------------------------
# CIFAR-10 class labels
# -----------------------------
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print(f"Predicted class: {classes[pred_class]} (index {pred_class})")