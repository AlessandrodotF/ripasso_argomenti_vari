# custom dataloader

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
import glob
import numpy as np
from typing import List, Tuple

# from torchmetrics.classification import (
#    MulticlassJaccardIndex,
#    MulticlassF1Score,
#    MulticlassAccuracy,
#    MeanIoU
# )
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassConfusionMatrix

# una cosa che potrei fare è considerare un problema sipo segmentation
# dove devo applicare trasfromazioni simultanee a img e mask:


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir_imgs: str,
        root_dir_lbls: str,
        annotations: List[Tuple[str, str]],
        augment: bool = False,
    ):
        self.augment = augment
        self.annotations = annotations
        self.root_dir_imgs = root_dir_imgs
        self.root_dir_lbls = root_dir_lbls

        # SOLO train
        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
                ),  # normaliz. random, da scegliere conr criterio
                ToTensorV2(),  # non normalizza nulla, fa solo trasposizione. ToTensor() di orch invece normalizza anche
            ]
        )

        # VAL/TEST
        self.test_transform = A.Compose(
            [
                A.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
                ),  # normaliz. random, da scegliere conr criterio
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name, mask_name = self.annotations[idx]

        img_path = os.path.join(self.root_dir_imgs, img_name)
        mask_path = os.path.join(self.root_dir_lbls, mask_name)

        # apertura immagini come numpy array per Alb.
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.augment:
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.test_transform(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"].long()  # per CrossEntropyLoss

        return image, mask


def build_DataLoader(annotations: List[Tuple[str, str]], dir_imgs: str, dir_lbls: str):
    train_ann, test_ann = train_test_split(annotations, test_size=0.1, random_state=42)

    dataset_train = CustomDataset(
        annotations=train_ann,
        root_dir_imgs=dir_imgs,
        root_dir_lbls=dir_lbls,
        augment=True,
    )

    dataset_test = CustomDataset(
        annotations=test_ann,
        root_dir_imgs=dir_imgs,
        root_dir_lbls=dir_lbls,
        augment=False,
    )

    loader_train = DataLoader(
        dataset_train, num_workers=4, pin_memory=True, batch_size=32, shuffle=True
    )
    loader_test = DataLoader(
        dataset_test, num_workers=4, pin_memory=True, batch_size=32, shuffle=False
    )

    return loader_train, loader_test


root_dir = "data"
dir_imgs = os.path.join(root_dir, "fake_imgs")
dir_lbls = os.path.join(root_dir, "fake_lbls")
list_lbls = [
    os.path.basename(file) for file in glob.glob(os.path.join(dir_lbls, "*.jpg"))
]
list_imgs = [
    os.path.basename(file) for file in glob.glob(os.path.join(dir_imgs, "*.jpg"))
]

annotations = list(zip(list_imgs, list_lbls))
# esempio del rigo prima
annotations = [
    ("img1.jpg", "lbl1.jpg"),
    ("img2.jpg", "lbl2.jpg"),
    ("img3.jpg", "lbl3.jpg"),
    ("img4.jpg", "lbl4.jpg"),
]
loader_train, loader_test = build_DataLoader(annotations, dir_imgs, dir_lbls)

model = ...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
n_epochs = 5
model.train()

for epoch in range(1, n_epochs + 1):
    loss_total = 0.0

    for images, lbls in loader_train:
        images = images.to(device)
        lbls = lbls.to(device).long()

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, lbls)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
    print(f"Loss in epoch {epoch} : {loss_total / len(loader_train)}")

NUM_CLASSES = 10

metrics = {
    "mIoU": MeanIoU(
        num_classes=NUM_CLASSES, include_background=False, per_class=True
    ).to(device),
}


# confmat = MulticlassConfusionMatrix(NUM_CLASSES, ignore_index=0).to(device)

model.eval()
with torch.no_grad():
    for images, labels in loader_test:
        images = images.to(device)
        labels = labels.to(device)

        preds = torch.argmax(model(images), dim=1)

        for m in metrics.values():
            m.update(preds, labels)

        # miou_per_class.update(preds, labels)
        # confmat.update(preds, labels)

# scalari
for name, m in metrics.items():
    print(f"{name}: {m.compute().item():.4f}")
    m.reset()

# per classe
iou_classes = miou_per_class.compute()
for i, val in enumerate(iou_classes):
    print(f"Class {i} IoU: {val:.4f}")
miou_per_class.reset()

# confusion matrix
cm = confmat.compute()
confmat.reset()
print("Confusion matrix:\n", cm)
