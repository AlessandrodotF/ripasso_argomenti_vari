import torch
from torch import randint
from torchmetrics.segmentation import MeanIoU

# The metric is optimal at a value of 1 and worst at a value of 0, -1 is returned
# if class is completely absent both from prediction and the ground truth labels.
miou = MeanIoU()
preds = randint(0, 1, (10, 3, 128, 128), generator=torch.Generator().manual_seed(42))
target = randint(0, 1, (10, 3, 128, 128), generator=torch.Generator().manual_seed(43))


miou = MeanIoU(
    num_classes=3, per_class=True, include_background=True, input_format="index"
)
# print(miou(preds, target))


from torchmetrics import Accuracy, F1Score, Recall

preds = randint(0, 9, (10, 4, 1), generator=torch.Generator().manual_seed(42))
target = randint(0, 9, (10, 4, 1), generator=torch.Generator().manual_seed(43))

print(preds)
metric = Accuracy(task="multiclass", num_classes=10)
print(metric(preds, target))
