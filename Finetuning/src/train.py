from cProfile import label
from importlib.resources import path
from hydra import main
from omegaconf import DictConfig
from omegaconf import OmegaConf
from utils import dice_loss_with_logits
from settings import * 

@main(config_path="configs", config_name="PASTIS_LIANet")
def main_cfg(args: DictConfig):
    # only one visible device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from torchmetrics import MetricTracker, MetricCollection
    from metrics import multiclass_segmentation_metrics, regression_metrics
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter

    from helpers import load_train_eval_datasets, load_model_class
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

    
    # COMPLETE_TILESIZE = area[args.checkpoint_area.split("_")[0]] if args.task not in ["BurnScars", "PASTIS_T31TFM", "PASTIS_T32ULU"] else 10980
    COMPLETE_TILESIZE = 10980
    TASK_TYPE = "regression" if args.task in ["meta_canopy_height", "building_footprints"] else "segmentation"

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # model_size_tag = args.checkpoint_area.split("_")[1]
    model_size_tag = args.checkpoint_area.split("_")[0] if args.checkpoint_area.split("_")[0] in ["5k", "7k", "10k"] else "full_tile"


    # make consisting nameing of the folders
    if args.model_type == "unet":
        if args.val_folds != "None": model_name = f"unet_valFolds{args.val_folds[0]}"
        else: model_name = f"unet_full_tile_nonburned"
    elif args.model_type == "micro_unet":
        if args.val_folds != "None": model_name = f"micro_unet_valFolds{args.val_folds[0]}"
        else: model_name = f"micro_unet_full_tile_nonburned"
    elif args.model_type == "replace_final_block":
        if args.val_folds != "None": model_name = f"LIANet_valFolds{args.val_folds[0]}"
        else: model_name = f"LIANet_full_tile_nonburned"
    elif args.model_type == "replace_final_block_4x":
        model_name = f"replace_final_block__{model_size_tag}_backbone"
    else:
        raise NotImplementedError("Model naming not implemented for this model type")
    
    OUTPUTDIR = os.path.join(args.logging_directory,
                             args.task,
                            #  args.checkpoint_area.split("_")[0],
                             model_name,
                             now)
    
    os.makedirs(OUTPUTDIR, exist_ok=True)

    # drop the config file to outputdir as json
    with open(os.path.join(OUTPUTDIR, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(args, resolve=True), f, indent=4)

    # ================= LOAD DATASET =================

    train_ds, val_ds = load_train_eval_datasets(
        task=args.task,
        TOP_DIR=TOP_DIR[args.task],
        S2_TILES=s2_tiles[args.task],
        LABELS=labels[args.task],
        train_area_bounds=args.train_area_bounds,
        COMPLETE_TILESIZE=COMPLETE_TILESIZE,
        exclude_px1_px2=(args.exclude_px1, args.exclude_px2) if args.task == "building_footprints" else None,
        val_folds=args.val_folds if args.task in ["PASTIS_T31TFM", "PASTIS_T32ULU", "PASTIS_T30UXV", "PASTIS_T31TFJ"] else None,
    )


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

    model = load_model_class(
        task=args.task,
        model_type=args.model_type, 
        MODEL_PATH=models[args.checkpoint_area], 
        NUM_CLASSES=num_classes[args.task], 
        ACTIVATION_FUNCTION=activation_functions[args.task])
    
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
            if args.task in ["PASTIS_T31TFM", "PASTIS_T32ULU", "PASTIS_T30UXV", "PASTIS_T31TFJ"]:
                criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
            else: criterion = torch.nn.CrossEntropyLoss()
            
        else:
            class_weights = weights[args.task]
            class_weights_tensor = torch.tensor(class_weights).cuda()
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        if not TASK_TYPE == "segmentation":
            raise ValueError("Cross Entropy loss can only be used for segmentation tasks")

    # LR Scheduler (Cosine Decay with Warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learningrate)

    # min_lr = 0.01*args.learningrate
    min_lr = 1e-6
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
            num_classes=num_classes[args.task],
            ignore_index=255 if args.task in ["PASTIS_T31TFM", "PASTIS_T32ULU", "PASTIS_T30UXV", "PASTIS_T31TFJ"] else None
        )

    metrics = MetricCollection(list_of_metrics).cuda()
    metrictracker = MetricTracker(
        metrics,
        maximize=maximize_list,
    )

    # ================= SETUP TENSORBOAR =================

    writer = SummaryWriter(log_dir=OUTPUTDIR)

    # ================= TRAINING LOOP HELPER FUNCTION =================

    def forward_model(model, model_type, batch, region_idx):

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
            # timestamp = batch["timestamp"] # This has changed in the new model
            timestamp = batch["delta_days"]
            x_s2 = batch["x_s2"]
            y_s2 = batch["y_s2"]
            label = batch["label"]

            timestamp = timestamp.cuda()
            x_s2 = x_s2.cuda()
            y_s2 = y_s2.cuda()
            label = label.cuda()
            
            assert timestamp.ndim == x_s2.ndim == y_s2.ndim == 1
            reconstruction, outputs = model(timestamp, x_s2, y_s2, region_idx)

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
            # region_list: {0: ["T31TFJ"], 1: ["T32ULU"], 2: ["T31TFM"], 3: ["T30UXV"]}
            region_idx = 0 if args.task == "PASTIS_T31TFJ" else 1 if args.task == "PASTIS_T32ULU" else 2 if args.task == "PASTIS_T31TFM" else 3
            _ , outputs, label = forward_model(model, args.model_type, batch, region_idx)
            if not torch.isfinite(outputs).all().item():
                print("Non-finite outputs")
                break
            # ANYWAYS: comput loss and backprop 
            # train_loss = bce(outputs, label.float()) + dice_loss_with_logits(outputs, label)
            if torch.isnan(outputs).any():
                print("NaNs in outputs"); break
            if torch.isnan(label).any():
                print("NaNs in label"); break
            train_loss = criterion(outputs.float(), label.long())
            if not torch.isfinite(train_loss).item():
                print("Non-finite loss:", train_loss)
                break
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

                    _ , outputs, label = forward_model(model, args.model_type, batch, region_idx)

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
                    elif args.task in ["PASTIS_T31TFM", "PASTIS_T32ULU", "PASTIS_T30UXV", "PASTIS_T31TFJ"]:
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
                    
                    cmap = ListedColormap(colors)

                counter = 0
                for batch in tqdm(validation_dataloader,total=10,desc=f"Epoch {epoch+1}/{args.epochs} - Plotting"):
                    
                    reconstruction, outputs, label = forward_model(model, args.model_type, batch, region_idx)  

                    if args.model_type in ["unet", "micro_unet"]:
                        reconstruction = torch.zeros_like(batch["s2data"])
                    if args.task == "BurnScars":
                        burned_pixel_count = batch["burned_pixel_count"]
                        
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
                        if args.task == "BurnScars": axs[3].set_title(f'Label with burn: {burned_pixel_count[0].detach().cpu()}%')
                        else: axs[3].set_title("Label")
                        

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
        ckpt = {
                "epoch": epoch,
                "globalstep": globalstep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "args": dict(args) if hasattr(args, "keys") else None,
            }
            
        torch.save(ckpt, os.path.join(OUTPUTDIR, "last.pt"))

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