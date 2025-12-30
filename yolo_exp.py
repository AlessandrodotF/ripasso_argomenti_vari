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

# in generale le metriche importanti, ANCHE PER UNO SPECIFICO PROBLEM DI OBJECT DETECTION, sono:
# 1) IoU
# 2) average precision (AP) : calcola l area sotto la precision recall curve. Un singolo valoro per
# quantificare le performance del modello dal punto di vista di precision e recall
# 3) Mean average precision (mAP): extend the concept of average precision
# su questa scia è utile tracciare anche  Precision,Recall,F1
# ##############################################################
# training da zero:
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="dataset.yaml", epochs=50)
metrics = model.val()  # no arguments needed, dataset and settings remembered


# che sia train, fine tune, o usare il modello cosi com'è
# .val() calcola

# Per Object Detection
# mAP@0.5 mAP@0.5:0.95 Precision Recall F1-score Metriche per classe Confusion matrix PR curves


# suppongo di avere 3 casi:


# DATA AUGMENTATION: LA BEST PRACTICE È USRE UELLE NATIVE CIOÈ
model.train(data="coco.yaml", epochs=100, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

# ampiezzo massime di jitter
hsv_h = 0.03  # hue - tonalità(colore)
hsv_s = 0.6  # saturation - saturazione
hsv_v = 0.5  # luminosity - luminosità

# probabilità
fliplr = 0.5  # flip orizzontale (conviene usarlo se sinistra/destra non conta)
erasing = 0.4  # cancella rettangoli casuali
mixup = 0.2  # fonde due immagini insieme (non mischiare valore troppo alti con mosaic)

# range di scaling [1 - scale, 1 + scale]
scale = 0.5  # (zoom in/out)

# YOLO sposta l’immagine fino al ±10% di larghezza/altezza.
translate = 0.1  # shift immagine (rende il modello meno sensibile alla posizione dell immagine )

# p ∈ [-perspective, +perspective]
perspective = 0.0005  # Trasformazione prospettica 3D leggera, utile con droni, cctv, camere grandalogari

# È una probabilità per batch, non per singola immagine.
mosaic = 1.0  # unisce 4 immagini in una sola (STRA IMPORTANTE)


auto_augment = "randaugment"  # Applica automaticamente policy di augmentation.

# mosaic, hsv_*, fliplr sono fondamentali
