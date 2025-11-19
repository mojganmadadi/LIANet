from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def regression_metrics():
    list_of_metrics = [
        MeanAbsoluteError(),
        MeanSquaredError(),
        PeakSignalNoiseRatio(data_range=1.0),
        StructuralSimilarityIndexMeasure(data_range=1.0)
    ]
    
    maximize_list = [
        False,  # MAE (lower better)
        False,  # MSE (lower better)
        True,   # PSNR (higher better)
        True,   # SSIM (higher better)
    ]

    return list_of_metrics, maximize_list