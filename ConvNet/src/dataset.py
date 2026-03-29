from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

IMG_SIZE = 32  # CIFAR original size

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            class_dir = self.root_dir / cls_name
            for image_path in class_dir.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.image_paths.append(image_path)
                    self.labels.append(self.class_to_idx[cls_name])

        # Transform: Resize, optional light online augmentation, ToTensor, Normalize
        transforms.Compose([
            transforms.Resize(36),                  # make it bigger first
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # crop back to 32x32
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Warning: corrupted image skipped: {image_path}")
            return self.__getitem__((idx + 1) % len(self))

        # Label as long/int for classification
        label = torch.tensor(label, dtype=torch.long)

        return image, label