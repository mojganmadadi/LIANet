from torchmetrics.classification import (
    MulticlassJaccardIndex,       # IoU
    MulticlassAccuracy,           # Pixel accuracy
    MulticlassF1Score,            # F1 / Dice
    MulticlassPrecision,          # Precision
    MulticlassRecall,             # Recall
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def multiclass_segmentation_metrics(num_classes: int, ignore_index: int = None):
    """
    Returns a comprehensive set of metrics for multiclass segmentation,
    including both macro and micro averaged variants.
    """

    metrics_dict = {
        # --- Macro (per-class, equal weight) ---
        "jaccard_macro":   MulticlassJaccardIndex(num_classes=num_classes,ignore_index=ignore_index, average='macro'),
        "accuracy_macro":  MulticlassAccuracy(num_classes=num_classes,ignore_index=ignore_index, average='macro'),
        "f1_macro":        MulticlassF1Score(num_classes=num_classes,ignore_index=ignore_index, average='macro'),
        "precision_macro": MulticlassPrecision(num_classes=num_classes,ignore_index=ignore_index, average='macro'),
        "recall_macro":    MulticlassRecall(num_classes=num_classes,ignore_index=ignore_index, average='macro'),

        # --- Micro (global aggregation) ---
        "jaccard_micro":   MulticlassJaccardIndex(num_classes=num_classes,ignore_index=ignore_index, average='micro'),
        "accuracy_micro":  MulticlassAccuracy(num_classes=num_classes, ignore_index=ignore_index, average='micro'),
        "f1_micro":        MulticlassF1Score(num_classes=num_classes, ignore_index=ignore_index, average='micro'),
        "precision_micro": MulticlassPrecision(num_classes=num_classes, ignore_index=ignore_index, average='micro'),
        "recall_micro":    MulticlassRecall(num_classes=num_classes, ignore_index=ignore_index, average='micro'),
    }

    maximize_list = [
        True, True, True, True, True,  # macro
        True, True, True, True, True,  # micro
    ]

    return metrics_dict, maximize_list


def regression_metrics():
    """
    Returns a set of regression metrics with unique names and corresponding maximize list.
    """

    metrics_dict = {
        "mae":  MeanAbsoluteError(),
        "mse":  MeanSquaredError(),
    }

    maximize_list = [
        False,  # MAE (lower better)
        False,  # MSE (lower better)
    ]

    return metrics_dict, maximize_list
