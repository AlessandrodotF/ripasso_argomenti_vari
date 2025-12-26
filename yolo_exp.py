# https://github.com/ultralytics/ultralytics


# fine tuning yolo (il classificatore originale viene
# automaticamente sostituito in base al numeor di classi che passo)


################################################
# Questo è per il fine tuning, carico i pesi del modello
# e dataset.yaml ha la forma

# path: /percorso/dataset
# train: images/train
# val: images/val
#
# nc: 2
# names: ["gatto", "cane"]

# DOVE
# dataset/
# ├── images/
# │   ├── train/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── val/
# │       └── img3.jpg
# ├── labels/
# │   ├── train/
# │   │   ├── img1.txt #stesso nome immagine
# │   │   └── img2.txt
# │   └── val/
# │       └── img3.txt
# └── dataset.yaml

# generico contenuto label
# una riga = un oggetto <class_id> <x_center centro bbox (0–1)> <y_center centro bbox (0–1)> <width centro bbox (0–1)> <height centro bbox (0–1)>

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="dataset.yaml", epochs=50)

# supporta tutta una serie di argoment come:
# FP16 (light compression), plots (save PR curves, confmatrix etc)
metrics = model.val()  # no arguments needed, dataset and settings remembered
print(metrics.confusion_matrix.to_df())


# What metrics can I get from YOLO11 model validation?
# mAP50 (mean Average Precision at IoU threshold 0.5)
# mAP75 (mean Average Precision at IoU threshold 0.75)
# mAP50-95 (mean Average Precision across multiple IoU thresholds from 0.5 to 0.95)


################################################################
# training da zero:
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="dataset.yaml", epochs=50)
metrics = model.val()  # no arguments needed, dataset and settings remembered
