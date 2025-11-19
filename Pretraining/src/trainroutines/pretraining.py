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


        self.globalstep = 0
        self.loss = 0
        self.model = self.model
        self.tracker = tracker

    def fit(self):
        
        self.current_epoch = 1

        self.model.train() 
        for epoch in range(self.current_epoch, self.events.nEpochs + 1):
            self.tb_writer.add_scalar(f"lr/over_epoch", self.optimizer.param_groups[0]["lr"], global_step=self.current_epoch)
            self._train_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()

            if self.current_epoch % self.validation_epochs == 0:
                self.model.eval()
                self._plot()         
                self._save_checkpoint("latest_validation_checkpoint")  
                self.model.train() 
            self.current_epoch += 1
        self.tb_writer.flush()

        return None
   
    def _train_one_epoch(self):
         
        pbar_train = tqdm(total=len(self.training_dataloader), desc=f"EPOCH: {self.current_epoch}",leave=False)
        
        runtime_buffer = deque(maxlen=10)
        running_mean_counter = 0
        
        for batch_idx, batch in enumerate(self.training_dataloader, start=1): 
            
            t0 = time.time()
            self._train_one_batch(batch)
            t1 = time.time()
            runtime_buffer.append(t1 - t0)
            running_mean_counter += 1

            if running_mean_counter >= 20:
                if self.globalstep % 20 == 0:
                    self.tb_writer.add_scalar(f"speed/mean_itter_time", np.mean(list(runtime_buffer)), global_step=self.globalstep)

            if self.globalstep % 20 == 0:
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
                reserved = torch.cuda.memory_reserved(device) / 1024**2    # MB
                self.tb_writer.add_scalar(f"gpu/allocated", allocated, global_step=self.globalstep)
                self.tb_writer.add_scalar(f"gpu/reserved", reserved, global_step=self.globalstep)

            pbar_train.update()

        pbar_train.close()        
        
        return None

    @abstractmethod
    def _train_one_batch(self, batch):
        x_s2 = batch["x_s2"]
        y_s2 = batch["y_s2"]
        s2data = batch["s2data"]
        timestamp = batch["timestamp"]
        

        x_s2 = x_s2.cuda(self.rank)
        y_s2 = y_s2.cuda(self.rank)
        s2data = s2data.cuda(self.rank)
        timestamp = timestamp.cuda(self.rank)
        prediction = self.model(timestamp, x_s2, y_s2)

        self.loss = self.lossfunction(prediction, s2data)
        self.optimizer.zero_grad()
        self.loss.backward()

        clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.globalstep += 1

        if self.globalstep % 10 == 0:
            self.tb_writer.add_scalar("train/loss", self.loss.item(), self.globalstep)
        return


    @abstractmethod
    def _plot(self,task:str="regression"):
        with torch.no_grad():

            for batch_idx, batch in enumerate(self.plot_dataloader):

                x_s2 = batch["x_s2"].cuda(self.rank)
                y_s2 = batch["y_s2"].cuda(self.rank)
                s2data = batch["s2data"].cuda(self.rank)
                s2data_np =s2data.detach().cpu().numpy()

                B = x_s2.size(0)  
                T = 4  

                predictions = []

                for t in range(T):
                    timestamp_batch = torch.full((B,), t, dtype=torch.long, device=f"cuda:{self.rank}")
                    pred = self.model(timestamp_batch, x_s2, y_s2)
                    predictions.append(pred)

                predictions = torch.stack(predictions, dim=0).detach().cpu().numpy()  # (4, B, C, H, W)

                for ijk in range(min(10, B)):

                    s2_img_predicted_0 = predictions[0, ijk]
                    s2_img_predicted_1 = predictions[1, ijk]
                    s2_img_predicted_2 = predictions[2, ijk]
                    s2_img_predicted_3 = predictions[3, ijk]

                    s2data_0_np = s2data_np[ijk, 0]  # winter
                    s2data_1_np = s2data_np[ijk, 1]  # spring
                    s2data_2_np = s2data_np[ijk, 2]  # summer
                    s2data_3_np = s2data_np[ijk, 3]  # autumn

                    fig, ax = plt.subplots(2, 4, figsize=(15, 10))

                    ax[0, 0].imshow(s2_to_rgb(s2data_0_np))
                    ax[0, 0].set_title(f"GTWinter")
                    ax[0, 1].imshow(s2_to_rgb(s2data_1_np))
                    ax[0, 1].set_title(f"GTSpring")
                    ax[0, 2].imshow(s2_to_rgb(s2data_2_np))
                    ax[0, 2].set_title(f"GTSummer")
                    ax[0, 3].imshow(s2_to_rgb(s2data_3_np))
                    ax[0, 3].set_title(f"GTAutumn")
                    ax[1, 0].imshow(s2_to_rgb(s2_img_predicted_0))
                    ax[1, 0].set_title(f"PredWinter")
                    ax[1, 1].imshow(s2_to_rgb(s2_img_predicted_1))
                    ax[1, 1].set_title(f"PredSpring")
                    ax[1, 2].imshow(s2_to_rgb(s2_img_predicted_2))
                    ax[1, 2].set_title(f"PredSummer")
                    ax[1, 3].imshow(s2_to_rgb(s2_img_predicted_3))
                    ax[1, 3].set_title(f"PredAutumn")
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

        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.globalstep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            },
           outputloc)
          
        return None
              
    def finalize(self):

        self.tb_writer.close()
        
        self._save_checkpoint()
        
        return None
