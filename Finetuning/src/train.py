from cProfile import label
from hydra import main
from omegaconf import DictConfig
from omegaconf import OmegaConf
from utils import dice_loss_with_logits
from settings import * 

@main(config_path="configs", config_name="BF_clas_LIANet")
def main_cfg(args: DictConfig):
    # only one visible device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from torchmetrics import MetricTracker, MetricCollection
    from metrics import multiclass_segmentation_metrics, regression_metrics
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter

    from datasets import DynamicWorld, MetaCanopyHeights, DominantLeafTypeSegmentation, BuildingCoverageRaster, BuildingBinaryRaster
    from models.models_finetune import DownstreamModel, UNet, MicroUNet

    from utils import s2_to_rgb
    from lr_scheduler import CosineAnnealingWarmupLR

    from datetime import datetime
    from tqdm import tqdm
    import numpy as np
    import matplotlib
    import json

    # ajust backend
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # ================= FIX SEED =================
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   
    # ================= OPTIONAL CONSTANTS =================

    LABELS = labels[args.task]
    MODEL_PATH = models[args.checkpoint_area]
    COMPLETE_TILESIZE = area[args.checkpoint_area.split("_")[0]]
    NUM_CLASSES = num_classes[args.task]
    TASK_TYPE = "regression" if args.task in ["meta_canopy_height", "building_footprints"] else "segmentation"
    ACTIVATION_FUNCTION = activation_functions[args.task]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_size_tag = args.checkpoint_area.split("_")[1]
    
    # make consisting nameing of the folders
    if args.model_type == "unet":
        model_name = "unet"
    elif args.model_type == "micro_unet":
        model_name = "micro_unet"
    elif args.model_type == "replace_final_block":
        model_name = f"replace_final_block__{model_size_tag}_backbone"
    elif args.model_type == "replace_final_block_4x":
        model_name = f"replace_final_block__{model_size_tag}_backbone"
    else:
        raise NotImplementedError("Model naming not implemented for this model type")
    
    OUTPUTDIR = os.path.join(args.logging_directory,
                             args.task,
                             args.checkpoint_area.split("_")[0],
                             model_name,
                             now)
    
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # drop the config file to outputdir as json
    with open(os.path.join(OUTPUTDIR, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(args, resolve=True), f, indent=4)

    # ================= LOAD DATASET =================

    if args.task == "dynamic_world":
        train_ds = DynamicWorld(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = DynamicWorld(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )

    elif args.task == "meta_canopy_height":
        train_ds = MetaCanopyHeights(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = MetaCanopyHeights(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )

    elif args.task == "building_footprints":
        px1 = (0, 4800) # We lack labels for this area, so we exclude it from training/validation
        px2 = (3040, 6080)
        train_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )
        val_ds = BuildingCoverageRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )

    elif args.task == "building_footprints_binary":
        px1 = (0, 4800) # We lack labels for this area, so we exclude it from training/validation
        px2 = (3040, 6080)
        train_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )
        val_ds = BuildingBinaryRaster(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
            exclude_px1_px2=(px1, px2),
        )

    elif args.task == "dominant_leaf_type":
        train_ds = DominantLeafTypeSegmentation(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="train",
            complete_tile_size=COMPLETE_TILESIZE,
        )
        val_ds = DominantLeafTypeSegmentation(
            top_dir=TOP_DIR,
            s2_tiles=S2_TILES,
            labels=LABELS,
            training_bounds_left_top_right_bottom=args.train_area_bounds,
            train_val_key="val",
            complete_tile_size=COMPLETE_TILESIZE,
        )

    else:
        raise ValueError("Invalid task")


    # ================= GENERATE DATALOADERS =================

    training_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    persistent_workers=True if args.num_workers > 0 else False,
    prefetch_factor=16 if args.num_workers > 0 else None,
    )  

    # ================= BUILD MODEL =================

    if args.model_type in ["replace_final_block", "replace_final_block_4x"]:
        
        if args.task == "building_footprints_binary":
            if not args.model_type == "replace_final_block_4x":
                raise ValueError("Footprint classification must be run with 4x model")
        
        model = DownstreamModel(
            model_path=MODEL_PATH,
            checkpoint_path_relative="model_checkpoints/latest_validation_checkpoint.pt",
            adaption_strategy=args.model_type,
            num_classes=NUM_CLASSES,
            activation=ACTIVATION_FUNCTION,
        )

    elif args.model_type == "unet":

        if args.task == "building_footprints_binary":
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

    elif args.model_type == "micro_unet":
        if args.task == "building_footprints_binary":
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

    summary(model)

    model = model.cuda()

    # ================= LOSS FUNCTIONS, LR SCHEDULER AND OPTIMIZER =================

    if args.lossfunction == "l1":  
        criterion = torch.nn.L1Loss()
        if not TASK_TYPE == "regression":
            raise ValueError("L1 loss can only be used for regression tasks")
    elif args.lossfunction == "mse":
        criterion = torch.nn.MSELoss()
        if not TASK_TYPE == "regression":
            raise ValueError("MSE loss can only be used for regression tasks")
    elif args.lossfunction == "huber":
        criterion = torch.nn.HuberLoss(delta=0.1)
        if not TASK_TYPE == "regression":
            raise ValueError("Huber loss can only be used for regression tasks")
    elif args.lossfunction == "cross_entropy":  
        if args.weightedSegmentation == False:
            # bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device='cuda'))

            criterion = torch.nn.CrossEntropyLoss()
        else:
            class_weights = weights[args.task]
            class_weights_tensor = torch.tensor(class_weights).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        if not TASK_TYPE == "segmentation":
            raise ValueError("Cross Entropy loss can only be used for segmentation tasks")

    # LR Scheduler (Cosine Decay with Warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningrate)

    min_lr = 0.01*args.learningrate
    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr=min_lr,
    )

    # ================= METRICS =================

    if TASK_TYPE == "regression":
        list_of_metrics, maximize_list = regression_metrics()

    else:  # segmentation
        list_of_metrics, maximize_list = multiclass_segmentation_metrics(
            num_classes=NUM_CLASSES,
        )

    metrics = MetricCollection(list_of_metrics).cuda()
    metrictracker = MetricTracker(
        metrics,
        maximize=maximize_list,
    )

    # ================= SETUP TENSORBOAR =================

    writer = SummaryWriter(log_dir=OUTPUTDIR)

    # ================= TRAINING LOOP HELPER FUNCTION =================

    def forward_model(model, model_type, batch):

        # EITHER: classicl model like unet or so
        if args.model_type in ["unet", "micro_unet"]:
            
            s2 = batch["s2data"]
            label = batch["label"]

            s2 = s2.cuda()
            label = label.cuda()    

            outputs = model(s2)

            reconstruction = s2

        # OR: our fancy super cool model witch is way better
        else:   
            timestamp = batch["timestamp"]
            x_s2 = batch["x_s2"]
            y_s2 = batch["y_s2"]
            label = batch["label"]

            timestamp = timestamp.cuda()
            x_s2 = x_s2.cuda()
            y_s2 = y_s2.cuda()
            label = label.cuda()
            
            assert timestamp.ndim == x_s2.ndim == y_s2.ndim == 1
            
            reconstruction, outputs = model(timestamp, x_s2, y_s2)

        return reconstruction, outputs, label

    # ================= TRAINING LOOP =================

    train_loss = 0.0
    globalstep = 0

    for epoch in range(0,args.epochs):

        # =================================================
        # TRAINING
        # =================================================

        model.train()
        for batch in tqdm(training_dataloader,total=len(training_dataloader),desc=f"Epoch {epoch+1}/{args.epochs} - Training"):

            optimizer.zero_grad()

            _ , outputs, label = forward_model(model, args.model_type, batch)

            # ANYWAYS: comput loss and backprop 
            # train_loss = bce(outputs, label.float()) + dice_loss_with_logits(outputs, label)
            train_loss = criterion(outputs.float(), label)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # logging
            writer.add_scalar("train/loss", train_loss.item(), globalstep)
            globalstep += 1

        # Step LR Scheduler
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch)
        # scheduler.step()

        # =================================================
        # VALIDATION
        # =================================================

        if epoch != 0 and epoch % args.validate_every_n_epochs == 0:

            model.eval()

            metrictracker.increment()

            with torch.no_grad():
                for batch in tqdm(validation_dataloader,total=len(validation_dataloader),desc=f"Epoch {epoch+1}/{args.epochs} - Validation"):

                    _ , outputs, label = forward_model(model, args.model_type, batch)

                    metrictracker.update(outputs, label)

            # save metrics to json and tensorboard
            metric_results = metrictracker.compute()
            for metric_name, metric_value in metric_results.items():
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)    


        # =================================================
        # PLOTTING
        # =================================================

        if epoch != 0 and epoch % args.plot_every_n_epochs == 0:

            model.eval()

            with torch.no_grad():

                # get the colormap for segmentation tasks
                if TASK_TYPE == "segmentation":

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
                    else:
                        colors = ["#FFFFFF", "#000000"]
                        vvmin, vvmax = 0, 1
                    
                    cmap = ListedColormap(colors)

                counter = 0
                for batch in tqdm(validation_dataloader,total=10,desc=f"Epoch {epoch+1}/{args.epochs} - Plotting"):
                    
                    reconstruction, outputs, label = forward_model(model, args.model_type, batch)  

                    if args.model_type in ["unet", "micro_unet"]:
                        reconstruction = torch.zeros_like(batch["s2data"])
                        
                    s2_image = batch["s2data"].detach().cpu().numpy().astype(np.float32)
                    reconstruction = reconstruction.detach().cpu().numpy().astype(np.float32)
                    outputs = outputs.detach().cpu().numpy()
                    label = label.detach().cpu().numpy()

                    fig, axs = plt.subplots(1, 4, figsize=(15, 5))


                    rgb0 = s2_to_rgb(s2_image[0]).astype(np.float32)
                    rgb1 = s2_to_rgb(reconstruction[0]).astype(np.float32)
                    axs[0].imshow(rgb0)
                    axs[0].set_title("Input")

                    axs[1].imshow(rgb1)
                    axs[1].set_title("Reconstruction")
    
                    if TASK_TYPE == "regression":

                        axs[2].imshow(outputs[0,0],vmin=0,vmax=1)
                        axs[2].set_title("Outputs")

                        axs[3].imshow(label[0,0],vmin=0,vmax=1)
                        axs[3].set_title("Label")

                    else:
                        pred_argmax = np.argmax(outputs, axis=1).astype(np.uint8)
                        label_to_plot = label.astype(np.uint8)

                        axs[2].imshow(pred_argmax[0], cmap=cmap, vmin=vvmin, vmax=vvmax)
                        axs[2].set_title("Outputs")
                        axs[3].imshow(label_to_plot[0], cmap=cmap, vmin=vvmin, vmax=vvmax)
                        axs[3].set_title("Label")

                    plt.tight_layout()
                    # print(
                    #     "dtypes:",
                    #     "rgb_in", rgb0.dtype,
                    #     "rgb_rec", rgb1.dtype,
                    #     "pred", (pred_argmax.dtype if TASK_TYPE!="regression" else outputs.dtype),
                    #     "label", (label_to_plot.dtype if TASK_TYPE!="regression" else label.dtype),
                    #     )
                    writer.add_figure(f"val/visualization_{counter}", fig, epoch)
                    #plt.savefig("test.png")
                    plt.close()

                    counter += 1
                    if counter == 9:
                        break

        # =================================================
        # Before next epoch
        # =================================================
        writer.flush()


    # =================================================
    # Finsih Script
    # =================================================

    # find best set of metrics and save to 
    best_vals, best_steps = metrictracker.best_metric(return_step=True)
    with open(os.path.join(OUTPUTDIR, "best_metrics.json"), "w") as f:
        json.dump({"best_values": best_vals}, f, indent=4)

    with open(os.path.join(OUTPUTDIR, "best_steps.json"), "w") as f:
        json.dump({"best_values": best_steps}, f, indent=4)

    writer.close()


if __name__ == "__main__":
    main_cfg()