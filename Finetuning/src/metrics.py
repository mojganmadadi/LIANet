from torchmetrics.classification import (
    MulticlassJaccardIndex,       # IoU
    MulticlassAccuracy,           # Pixel accuracy
    MulticlassF1Score,            # F1 / Dice
    MulticlassPrecision,          # Precision
    MulticlassRecall,             # Recall
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def multiclass_segmentation_metrics(num_classes: int, ignore_index: int = None):
    """
    Returns a comprehensive set of metrics for multiclass segmentation,
    including both macro and micro averaged variants.
    """

    metric_kwargs = {
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "zero_division": 0,
    }

    metrics_dict = {
        # --- Macro (per-class, equal weight) ---
        "jaccard_macro":   MulticlassJaccardIndex(**metric_kwargs, average='macro'),
        "accuracy_macro":  MulticlassAccuracy(**metric_kwargs, average='macro'),
        "f1_macro":        MulticlassF1Score(**metric_kwargs, average='macro'),
        "precision_macro": MulticlassPrecision(**metric_kwargs, average='macro'),
        "recall_macro":    MulticlassRecall(**metric_kwargs, average='macro'),

        # --- Micro (global aggregation) ---
        "jaccard_micro":   MulticlassJaccardIndex(**metric_kwargs, average='micro'),
        "accuracy_micro":  MulticlassAccuracy(**metric_kwargs, average='micro'),
        "f1_micro":        MulticlassF1Score(**metric_kwargs, average='micro'),
        "precision_micro": MulticlassPrecision(**metric_kwargs, average='micro'),
        "recall_micro":    MulticlassRecall(**metric_kwargs, average='micro'),
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
        "pearson": PearsonCorrCoef(),
    }

    maximize_list = [
        False,  # MAE (lower better)
        False,  # MSE (lower better)
        True,  # Pearson correlation (higher better)
    ]

    return metrics_dict, maximize_list
