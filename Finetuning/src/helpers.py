from datasets import DynamicWorld, MetaCanopyHeights, DominantLeafTypeSegmentation, BuildingCoverageRaster, BuildingBinaryRaster, PASTIS, BurnScars
from models.models_finetune import DownstreamModel, UNet, MicroUNet


def load_train_eval_datasets(
    task, 
    TOP_DIR, 
    S2_TILES, 
    LABELS, 
    train_area_bounds, 
    COMPLETE_TILESIZE, 
    exclude_px1_px2=None,
    val_folds=None,):
    
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

    elif task == "building_footprints":
        px1 = (0, 4800) # We lack labels for this area, so we exclude it from training/validation
        px2 = (3040, 6080)
        train_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )
        val_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )

    elif task == "building_footprints_binary":
        px1 = (0, 4800) # We lack labels for this area, so we exclude it from training/validation
        px2 = (3040, 6080)
        train_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )
        val_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
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
    elif task == "PASTIS":
        train_ds = PASTIS(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="train",
            val_folds=val_folds,
        )
        val_ds = PASTIS(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            train_val_key="val",
            val_folds=val_folds,
        )
    elif task == "BurnScars":
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

def load_model_class(
    task, 
    model_type, 
    MODEL_PATH, 
    NUM_CLASSES, 
    ACTIVATION_FUNCTION):
    if model_type in ["replace_final_block", "replace_final_block_4x"]:
        
        if task == "building_footprints_binary":
            if not model_type == "replace_final_block_4x":
                raise ValueError("Footprint classification must be run with 4x model")
        
        model = DownstreamModel(
            model_path=MODEL_PATH,
            checkpoint_path_relative="model_checkpoints/latest_validation_checkpoint.pt",
            adaption_strategy=model_type,
            num_classes=NUM_CLASSES,
            activation=ACTIVATION_FUNCTION,
        )

    elif model_type == "unet":
        if task == "building_footprints_binary":
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
        if task == "building_footprints_binary":
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

    else:
        raise ValueError("Invalid model_type")
    
    return model