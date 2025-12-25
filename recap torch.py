# custom dataloader

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None, target_transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, label = self.annotations[idx]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def build_DataLoader(annotations):
    images = [a[0] for a in annotations]
    labels = [a[1] for a in annotations]

    train_img, test_img, train_lbl, test_lbl = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,  # cosi considero anche la distribuzione delle classi
    )

    train_ann = list(zip(train_img, train_lbl))
    test_ann = list(zip(test_img, test_lbl))

    my_transforms_train = transforms.Compose(
        [
            transforms.Resize((123, 123)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    my_transforms_test = transforms.Compose(
        [
            transforms.Resize((123, 123)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_train = CustomDataset(
        annotations=train_ann, img_dir="data/images", transform=my_transforms_train
    )

    dataset_test = CustomDataset(
        annotations=test_ann, img_dir="data/images", transform=my_transforms_test
    )

    loader_train = DataLoader(
        dataset_train, num_workers=4, pin_memory=True, batch_size=32, shuffle=True
    )
    loader_test = DataLoader(
        dataset_test, num_workers=4, pin_memory=True, batch_size=32, shuffle=False
    )

    return loader_train, loader_test


import glob
import os


dir_imgs = os.path.join("prova", "fake_imgs")
dir_lbls = os.path.koin("prova", "fake_lbls")

for file in glob.glob(os.path.join(dir_lbls, "*.txt")):
    print(os.path.basename(file))


annotations = [
    ("img1.jpg", 0),
    ("img2.jpg", 1),
    ("img3.jpg", 0),
]
# loader_train, loader_test = build_DataLoader(annotations)
