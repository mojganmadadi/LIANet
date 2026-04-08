from abc import ABC, abstractmethod
from collections import deque
import os
import time

import matplotlib
matplotlib.use("Agg")  # set backend before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils import s2_to_rgb


class LIANetTrainer:
    
    def __init__(
            self,
            model,
            training_dataset,
            training_dataloader,
            plot_dataset,
            plot_dataloader,
            optimizer,
            tracker,
            lossfunction,
            scheduler,
            rank,
            world_size,
            paths,
            events,
            tb_writer,
            validation_epochs):


        self.model = model
        self.training_dataset = training_dataset
        self.training_dataloader = training_dataloader
        self.plot_dataset = plot_dataset
        self.plot_dataloader = plot_dataloader 
        self.optimizer = optimizer
        self.tracker = tracker
        self.rank = rank
        self.world_size = world_size
        self.lossfunction = lossfunction
        self.scheduler = scheduler
        self.paths = paths
        self.events = events
        self.tb_writer = tb_writer
        self.validation_epochs = validation_epochs

        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.globalstep = 0
        self.loss = 0
        self.model = self.model
        self.tracker = tracker

    def fit(self):
        
        self.current_epoch = 1

        self.model.train() 
        for epoch in range(self.current_epoch, self.events.nEpochs + 1):

            if self.rank == 0:
                self.tb_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            self._train_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.current_epoch in self.events.special_save_Epochs:
                self._save_checkpoint()
            if self.current_epoch % self.validation_epochs == 0:
                self.model.eval()
                self._validate()
                self._plot()         
                self._save_checkpoint("latest_validation_checkpoint")  
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                self.model.train() 
            self.current_epoch += 1
        self.tb_writer.flush()

        return None
   
    def _train_one_epoch(self):
         
        pbar_train = tqdm(
            total=len(self.training_dataloader),
            desc=f"EPOCH: {self.current_epoch}",
            leave=False,
            disable=(self.rank != 0),
            position=0,
            dynamic_ncols=True,
        )
        
        runtime_buffer = deque(maxlen=10)
        running_mean_counter = 0
        # set epoch for proper shuffleing with DDP
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.training_dataloader.sampler.set_epoch(self.current_epoch)
        for batch_idx, batch in enumerate(self.training_dataloader, start=1): 
            
            if self.rank == 0:
                t0 = time.time()
            self._train_one_batch(batch)
            if self.rank == 0:
                t1 = time.time()
                runtime_buffer.append(t1 - t0)
                running_mean_counter += 1
                if running_mean_counter >= 20:
                    if self.globalstep % 20 == 0:
                        self.tb_writer.add_scalar(f"speed/mean_itter_time", np.mean(list(runtime_buffer)), global_step=self.globalstep)

                if self.globalstep % 500 == 0:
                    device = torch.cuda.current_device()
                    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
                    reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
                    self.tb_writer.add_scalar(f"gpu/allocated", allocated, global_step=self.globalstep)
                    self.tb_writer.add_scalar(f"gpu/reserved", reserved, global_step=self.globalstep)

            pbar_train.update(1)

        pbar_train.close()        
        
        return None

    @abstractmethod
    def _train_one_batch(self, batch):
        x_s2 = batch["x_s2"]
        y_s2 = batch["y_s2"]
        s2data = batch["s2data"]
        time_idx = batch["time_idx"]
        delta_days = batch["delta_days"]
        doy_sin = batch["doy_sin"]
        doy_cos = batch["doy_cos"]

        x_s2 = x_s2.to(self.rank, non_blocking=True).float()
        y_s2 = y_s2.to(self.rank, non_blocking=True).float()
        s2data = s2data.to(self.rank, non_blocking=True)
        time_idx = time_idx.to(self.rank, non_blocking=True)
        delta_days = delta_days.to(self.rank, non_blocking=True)
        doy_sin = doy_sin.to(self.rank, non_blocking=True)
        doy_cos = doy_cos.to(self.rank, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
            m = self.model.module if hasattr(self.model, "module") else self.model

            if m.time_mode == "index":
                prediction = self.model(time_idx, x_s2, y_s2)
            elif m.time_mode == "sinusoidal" or m.time_mode == "fourier_learned":
                prediction = self.model(delta_days, x_s2, y_s2)
            elif m.time_mode == "mlp":
                time_features = torch.stack([delta_days, doy_sin, doy_cos], dim=1)  # (B, 3)
                prediction = self.model(time_features, x_s2, y_s2)

            self.loss = self.lossfunction(prediction, s2data)

        self.scaler.scale(self.loss).backward()
        
        # unscale before clipping
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.globalstep += 1

        if self.globalstep % 10 == 0 and self.rank == 0:
            self.tb_writer.add_scalar("train/loss", self.loss.item(), self.globalstep)
        return


    def _validate(self):


        # evaluate data from old active windows
        self.tracker.increment()
        current_eval_cycle = self.tracker.n_steps - 1 # starts at one so set to zero
        with torch.no_grad():
            
            pbar_val = tqdm(
                total=len(self.plot_dataloader),
                desc=f"EPOCH: {self.current_epoch}",
                leave=False,
                disable=(self.rank != 0),
                dynamic_ncols=True,
                position=1,
            )
            pbar_val.set_description("validation")
            for batch_idx, batch in enumerate(self.plot_dataloader):
                x_s2 = batch["x_s2"]
                y_s2 = batch["y_s2"]
                s2data = batch["s2data"]
                time_idx = batch["time_idx"]
                delta_days = batch["delta_days"]
                doy_sin = batch["doy_sin"]
                doy_cos = batch["doy_cos"]

                x_s2 = x_s2.to(self.rank, non_blocking=True).float()
                y_s2 = y_s2.to(self.rank, non_blocking=True).float()
                s2data = s2data.to(self.rank, non_blocking=True)
                time_idx = time_idx.to(self.rank, non_blocking=True)
                delta_days = delta_days.to(self.rank, non_blocking=True)
                doy_sin = doy_sin.to(self.rank, non_blocking=True)
                doy_cos = doy_cos.to(self.rank, non_blocking=True)
                m = self.model.module if hasattr(self.model, "module") else self.model

                if m.time_mode == "index":
                    pred = self.model(time_idx, x_s2, y_s2)
                elif m.time_mode == "sinusoidal" or m.time_mode == "fourier_learned":
                    pred = self.model(delta_days, x_s2, y_s2)
                elif m.time_mode == "mlp":
                    time_features = torch.stack([delta_days, doy_sin, doy_cos], dim=1)  # (B, 3)
                    pred = self.model(time_features, x_s2, y_s2)
                # contiguous tensors
                prediction = pred.contiguous()
                s2data = s2data.contiguous()
                
                self.tracker.update(prediction, s2data)
                pbar_val.update(1)

            pbar_val.close()   


            results = self.tracker.compute_all()
            if self.rank == 0:
                for key, val in results.items():
                    self.tb_writer.add_scalar(f"val/{key}", val[current_eval_cycle], global_step=self.globalstep)
        return None
    
    @abstractmethod
    def _plot(self,task:str="regression"):

        # Only plot on main process in distributed mode
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return None

        with torch.no_grad():

            for batch_idx, batch in enumerate(self.plot_dataloader):
                x_s2 = batch["x_s2"]
                y_s2 = batch["y_s2"]
                s2data = batch["s2data"]
                time_idx = batch["time_idx"]
                date_str = batch["date_str"]
                delta_days = batch["delta_days"]
                doy_sin = batch["doy_sin"]
                doy_cos = batch["doy_cos"]

                x_s2 = x_s2.to(self.rank, non_blocking=True).float()
                y_s2 = y_s2.to(self.rank, non_blocking=True).float()
                s2data = s2data.to(self.rank, non_blocking=True)
                time_idx = time_idx.to(self.rank, non_blocking=True)
                delta_days = delta_days.to(self.rank, non_blocking=True)
                doy_sin = doy_sin.to(self.rank, non_blocking=True)
                doy_cos = doy_cos.to(self.rank, non_blocking=True)

                m = self.model.module if hasattr(self.model, "module") else self.model

                if m.time_mode == "index":
                    pred = self.model(time_idx, x_s2, y_s2)
                elif m.time_mode == "sinusoidal" or m.time_mode == "fourier_learned":
                    pred = self.model(delta_days, x_s2, y_s2)
                elif m.time_mode == "mlp":
                    time_features = torch.stack([delta_days, doy_sin, doy_cos], dim=1)  # (B, 3)
                    pred = self.model(time_features, x_s2, y_s2)
                

                B = x_s2.size(0)  

                predictions = pred.detach().float().cpu().numpy()  # (4, B, C, H, W)

                for ijk in range(min(10, B)):

                    s2_img_predicted = predictions[ijk]
                    fig, ax = plt.subplots(1,2, figsize=(15, 10))

                    gt_np = s2data[ijk].detach().float().cpu().numpy()
                    ax[0].imshow(s2_to_rgb(gt_np))
                    ax[0].set_title(f"GT x:{x_s2[ijk]} y:{y_s2[ijk]} t:{date_str[ijk]}")
                    ax[1].imshow(s2_to_rgb(s2_img_predicted))
                    ax[1].set_title(f"predicted")
                    for a in ax.flatten():
                        a.axis("off")

                    plt.tight_layout()
                    self.tb_writer.add_figure(f"example_output_{ijk}", fig, global_step=self.globalstep)
                    plt.close()

                break  # just one batch

        return None
    
    def _save_checkpoint(self,name_overwrite=None):
        
        if name_overwrite is None:
            outputloc =  os.path.join(self.paths.checkpoint_dir,f"checkpoint_Epoch{self.current_epoch}_Iteration{self.globalstep}.pt")
        else:                               
            outputloc = os.path.join(self.paths.checkpoint_dir,f"{name_overwrite}.pt")
        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save = self.model
        if hasattr(model_to_save, "module"):      # DDP
            model_to_save = model_to_save.module
        if hasattr(model_to_save, "_orig_mod"):   # torch.compile wrapper
            model_to_save = model_to_save._orig_mod
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.globalstep,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            },
           outputloc)
          
        return None
              
    def finalize(self):

        self.tb_writer.close()
        self._save_checkpoint()
        
        return None
