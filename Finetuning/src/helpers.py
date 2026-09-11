from datetime import datetime
import os

from datasets import DynamicWorld, MetaCanopyHeights, DominantLeafTypeSegmentation, BuildingCoverageRaster, BuildingBinaryRaster, PASTIS, BurnScars, AnnualEmbeddingDataset
from models.models_finetune import DownstreamModel, UNet, MicroUNet
from models.terratorch_models import TerraTorchFactorySegmentationModel


import numpy as np
import rasterio
from pathlib import Path

def compute_mean_std(tile_paths):
    """
    Compute per-band mean and std over a list of stacked Sentinel-2 GeoTIFF tiles.

    Parameters
    ----------
    tile_paths : list of str or Path
        List of file paths to stacked 12-band .tif images

    Returns
    -------
    mean : np.ndarray shape (12,)
    std  : np.ndarray shape (12,)
    """
    tiles_paths_list = [os.path.join(tile_paths, f) for f in os.listdir(tile_paths)]
    n_bands = 12

    # Running statistics
    pixel_count = np.zeros(n_bands, dtype=np.float64)
    band_sum = np.zeros(n_bands, dtype=np.float64)
    band_sum_sq = np.zeros(n_bands, dtype=np.float64)

    for tile in tiles_paths_list:
        with rasterio.open(tile) as src:
            data = src.read().astype(np.float64)  # shape: (12, H, W)

        # Flatten spatial dimensions
        data = data.reshape(n_bands, -1)

        # Mask NaNs if present
        valid_mask = ~np.isnan(data)

        for b in range(n_bands):
            band_pixels = data[b][valid_mask[b]]

            pixel_count[b] += band_pixels.size
            band_sum[b] += band_pixels.sum()
            band_sum_sq[b] += np.sum(band_pixels ** 2)

    mean = band_sum / pixel_count
    std = np.sqrt((band_sum_sq / pixel_count) - (mean ** 2))

    return mean, std

def load_train_eval_datasets(
    task, 
    TOP_DIR, 
    S2_TILES, 
    LABELS, 
    train_area_bounds, 
    COMPLETE_TILESIZE, 
    exclude_px1_px2=None,
    val_folds=None,
    args=None):

    input_source = getattr(args, "input_source", None)
    if input_source in {"alphaearth", "tessera"}:
        dataset_name, tile = _embedding_dataset_and_tile(args.task)
        source_cfg = getattr(args, input_source)
        manifest_path = _embedding_manifest_for_task(
            source_cfg.manifest_path,
            dataset_name,
            tile,
            input_source,
        )

        common = dict(
            lookup_path=source_cfg.lookup_path,
            manifest_path=manifest_path,
            dataset=dataset_name,
            tile=tile,
            task=args.task,
            input_source=input_source,
            val_folds=val_folds,
            cache_in_memory=getattr(source_cfg, "cache_in_memory", False),
        )
        train_ds = AnnualEmbeddingDataset(train_val_key="train", **common)
        val_ds = AnnualEmbeddingDataset(train_val_key="val", **common)
        return train_ds, val_ds

    if task == "dynamic_world":
        train_ds = DynamicWorld(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = DynamicWorld(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )

    elif task == "meta_canopy_height":
        train_ds = MetaCanopyHeights(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = MetaCanopyHeights(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )

    elif "BFPDensity" in task:
        train_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="train",
        )
        val_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="val",
        )

    elif "BFPBinary" in task:
        train_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="train",
        )
        val_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="val",
        )

    elif task == "dominant_leaf_type":
        train_ds = DominantLeafTypeSegmentation(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = DominantLeafTypeSegmentation(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )
    elif "PASTIS" in task:
        train_ds = PASTIS(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="train",
            val_folds=val_folds,
            cache_samples=getattr(args, "cache_samples", True),
            sample_cache_dir=getattr(args, "sample_cache_dir", None),
        )
        # all_tiles = {"T32ULU", "T31TFJ"}
        # val_tile = next(iter(all_tiles-{S2_TILES}))

        val_ds = PASTIS(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="val",
            val_folds=val_folds,
            cache_samples=getattr(args, "cache_samples", True),
            sample_cache_dir=getattr(args, "sample_cache_dir", None),
        )
    elif "BurnScars" in task:
        train_ds = BurnScars(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="train",
        )
        val_ds = BurnScars(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="val",
        )
    else:
        raise ValueError("Invalid task")
    return train_ds, val_ds


def _embedding_dataset_and_tile(task):
    """Map an embedding task name like ``PASTIS_joint_T31TFJ`` to (dataset, tile)."""
    tile = task.split("_")[-1]
    if "BurnScars" in task:
        return "HLS_BrunScars", tile
    if "PASTIS" in task:
        return "PASTIS", tile
    if "BFP" in task:
        return "BFP", tile
    raise ValueError(f"Unsupported annual embedding task: {task}")


def _embedding_manifest_for_task(configured_path, dataset_name, tile, input_source):
    """Resolve the per-tile embedding manifest for a task.

    The configured ``manifest_path`` names one tile, but cross-region evaluation needs the
    manifest of whichever tile is being read. Derive that filename from (dataset, tile) and
    require it to exist: silently falling back to the configured path would evaluate on the
    wrong tile's embeddings and report a plausible-looking but meaningless number.
    """
    manifest_path = Path(configured_path)
    expected_name = f"{dataset_name}_{tile}_{input_source}.parquet"
    candidate = manifest_path.with_name(expected_name)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Embedding manifest for {dataset_name}/{tile} ({input_source}) not found at "
            f"{candidate}. Cross-region evaluation requires the target tile's own manifest."
        )
    return str(candidate)


def load_model_class(
    task,
    model_type,
    MODEL_PATH,
    NUM_CLASSES,
    ACTIVATION_FUNCTION,
    TERRATORCH_CONFIG=None):
    if model_type in ["replace_final_block", "replace_final_block_4x"]:
        
        if "BFPBinary" in task:
            if not model_type == "replace_final_block_4x":
                raise ValueError("Footprint classification must be run with 4x model")
        
        model = DownstreamModel(
            model_path=MODEL_PATH,
            checkpoint_path_relative="model_checkpoints/latest_validation_checkpoint.pt",
            adaption_strategy=model_type,
            num_classes=NUM_CLASSES,
            # activation=ACTIVATION_FUNCTION,
        )

    elif model_type == "unet":
        if "BFPBinary" in task:
            model = UNet(n_channels=12,
                        n_classes=NUM_CLASSES,
                        backbone_size="small",
                        bilinear=True,
                        activation=ACTIVATION_FUNCTION,
                        upsample_4x=True)
        else:
            model = UNet(n_channels=12,
                        n_classes=NUM_CLASSES,
                        backbone_size="small",
                        bilinear=True,
                        activation=ACTIVATION_FUNCTION)

    elif model_type == "micro_unet":
        if "BFPBinary" in task:
            model = MicroUNet(n_channels=12,
                            num_classes=NUM_CLASSES,
                            bilinear=True,
                            activation=ACTIVATION_FUNCTION,
                            upsample_4x=True)
        else:
            model = MicroUNet(n_channels=12,
                            num_classes=NUM_CLASSES,
                            bilinear=True,
                            activation=ACTIVATION_FUNCTION,
                            upsample_4x=False)

    elif model_type == "terratorch_factory":
        model = TerraTorchFactorySegmentationModel(
            config=TERRATORCH_CONFIG,
            num_classes=NUM_CLASSES,
        )

    else:
        raise ValueError("Invalid model_type")
    
    return model

def provide_cmap(TASK_TYPE, args):
    if args.task == "dynamic_world":
        colors = [
                "#419BDF",  # water - blue
                "#397D49",  # trees - dark green
                "#88B053",  # grass - light green
                "#E4E8A1",  # crops - yellow-green
                "#E47474",  # built area - red
                "#A59B8F",  # bare ground - brown-gray
            ]
        vvmin, vvmax = 0, 5
    elif args.task == "dominant_leaf_type":
        colors = [
                "#FFFFFF",  # 0 - no data (white)
                "#4CAF50",  # 1 - broadleaf (green)
                "#1B5E20",  # 2 - needleleaf (dark green)
                ]
        vvmin, vvmax = 0, 2
    elif "PASTIS" in args.task:
        colors = [
                (0, 0, 0),
                (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
                (1.0, 0.4980392156862745, 0.054901960784313725),
                (1.0, 0.7333333333333333, 0.47058823529411764),
                (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
                (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                (1.0, 0.596078431372549, 0.5882352941176471),
                (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
                (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
                (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
                (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
                (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
                (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
                (1, 1, 1),
            ]
        vvmin, vvmax = 0, 19
    else:
        # colors = ["#FFFFFF", "#000000"]
        colors = ["#000000", "#FFFFFF"]

        vvmin, vvmax = 0, 1 
    return colors, vvmin, vvmax

def create_output_dir(args):
    if args.model_type == "unet":
        if "PASTIS" in args.task: 
            model_name = f"unet_valFolds{args.val_folds[0]}_lr{args.learningrate}_batchsize{args.batchsize}"
        else: 
            model_name = f"unet_lr{args.learningrate}_batchsize{args.batchsize}"
    elif args.model_type == "micro_unet":
        if "PASTIS" in args.task:
            model_name = f"micro_unet_valFolds{args.val_folds[0]}_lr{args.learningrate}_batchsize{args.batchsize}"
        else: 
            model_name = f"microunet_lr{args.learningrate}_batchsize{args.batchsize}"
    elif args.model_type == "replace_final_block":
        if "PASTIS" in args.task:
            model_name = f"LIANet_valFolds{args.val_folds[0]}_lr{args.learningrate}_batchsize{args.batchsize}"
        else: 
            model_name = f"LIANet_lr{args.learningrate}_batchsize{args.batchsize}"
    elif args.model_type == "replace_final_block_4x":
        model_name = f"replace_final_block_4x_lr{args.learningrate}_batchsize{args.batchsize}"
    else:
        raise NotImplementedError("Model naming not implemented for this model type")

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
    
    OUTPUTDIR = os.path.join(args.logging_directory,
                             args.task,
                             model_name,
                             now)
    return OUTPUTDIR