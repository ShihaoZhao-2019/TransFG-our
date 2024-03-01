'''
Function:
    define the cross entropy loss
Author:
    Charles
'''
import torch.nn.functional as F


'''binary cross entropy loss'''
def BinaryCrossEntropyLoss(preds, targets, scale_factor=1.0, size_average=True, loss_weight=None):
    if size_average:
        loss = F.binary_cross_entropy(preds, targets, reduction='mean') if loss_weight is None else \
               F.binary_cross_entropy(preds, targets, reduction='mean', weight=loss_weight)
    else:
        loss = F.binary_cross_entropy(preds, targets, reduction='sum') if loss_weight is None else \
               F.binary_cross_entropy(preds, targets, reduction='sum', weight=loss_weight)
    return loss * scale_factor