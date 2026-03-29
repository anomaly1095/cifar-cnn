from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms
import threading
from model import CIFAR10ConvNet10  # your ConvNet10

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 256
LR = 1e-3
WD = 1e-3
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded dataset path
BASE_DIR = Path(__file__).resolve().parent.parent  # points to CIFAR10/ConvNet
DATA_DIR = BASE_DIR.parent.parent / "datasets" / "cifar10_images_augmented"

# -----------------------------
# Accuracy function
# -----------------------------
def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum()
    return correct.float() / labels.size(0)

# -----------------------------
# Async save
# -----------------------------
def save_model_async(model, path):
    state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def _save():
        torch.save(state_dict, path)

    threading.Thread(target=_save, daemon=True).start()

# -----------------------------
# Training for one epoch
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)
        epoch_acc += accuracy(outputs, labels) * images.size(0)

    epoch_loss /= len(loader.dataset)
    epoch_acc /= len(loader.dataset)
    return epoch_loss, epoch_acc

# -----------------------------
# Validation
# -----------------------------
def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += accuracy(outputs, labels) * images.size(0)

    val_loss /= len(loader.dataset)
    val_acc /= len(loader.dataset)
    return val_loss, val_acc

# -----------------------------
# Main training loop
# -----------------------------
def main():
  # -----------------------------
  # Transforms
  # -----------------------------
  train_transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
  ])

  test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
  ])

  # -----------------------------
  # Datasets / loaders
  # -----------------------------
  train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
  val_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

  train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=4,
      persistent_workers=True,
      pin_memory=True
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=4,
      persistent_workers=True,
      pin_memory=True
  )

  # -----------------------------
  # Model / criterion / optimizer / scheduler
  # -----------------------------
  model = CIFAR10ConvNet10(num_classes=10).to(DEVICE)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.3
  )

  weights_dir = BASE_DIR / "weights"
  weights_dir.mkdir(exist_ok=True)

  best_val_acc = 0.0
  log_file = BASE_DIR / "checkpoints.log"

  # Write a separator line at the start of the log
  separator_line = "-" * 88 + "\n"
  with open(log_file, "a") as f:
    f.write(separator_line)
  
  # -----------------------------
  # Training loop
  # -----------------------------
  for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    scheduler.step(val_acc)

    log_line = (f"Epoch {epoch}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

    with open(log_file, "a") as f:
      f.write(log_line)

    # Save the best model
    if val_acc > best_val_acc:
      best_val_acc = val_acc
      best_model_path = weights_dir / "best_model.pth"
      save_model_async(model, best_model_path)
      print(f"Saved new best model with Val Acc: {best_val_acc:.4f} at {best_model_path}")

if __name__ == "__main__":
  main()