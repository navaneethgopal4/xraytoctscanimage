import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UnpairedDataset(Dataset):
    def __init__(self, data_dir, phase="train", transform=None):
        """
        Custom Dataset for unpaired image translation (CycleGAN).
        data_dir: base directory containing 'xray' and 'ct' folders.
        phase: 'train' or 'test'.
        """
        self.dir_A = os.path.join(data_dir, "xray", phase)
        self.dir_B = os.path.join(data_dir, "ct", phase)
        
        self.files_A = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.files_B = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
        
        if self.len_A == 0 or self.len_B == 0:
            raise RuntimeError(f"Error: Found 0 files in {self.dir_A} or {self.dir_B}. Please run download_data.py first.")

        # Default transforms if not provided
        if transform is None:
            if phase == "train":
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10), # Safe rotation, no vertical flip
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256), Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        path_A = self.files_A[index % self.len_A]
        # For unpaired dataset, select a random image from B for training
        # For test/eval, we can select deterministically
        idx_B = random.randint(0, self.len_B - 1)
        path_B = self.files_B[idx_B]
        
        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")
        
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        
        return {"A": item_A, "B": item_B, "path_A": path_A, "path_B": path_B}

    def __len__(self):
        return max(self.len_A, self.len_B)
