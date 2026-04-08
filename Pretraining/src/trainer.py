import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist

from datetime import datetime
from types import SimpleNamespace
import json
import os

import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection, MetricTracker
from torchinfo import summary

def ddp_setup(rank: int, world_size: int, port=None):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    if port is None:
        os.environ["MASTER_PORT"] = "12353"
    else:
        os.environ["MASTER_PORT"] = str(port)        
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return None


def setup_and_start_training(rank: int,
                             world_size: int,
                             config: DictConfig):
        
    if 0 in [int(x) for x in config.gpus]:
        port = 29501
    else:
        port = 29502
        
    ddp_setup(rank, world_size, port=port)

    if rank == 0:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        savepath = os.path.join(config.outputpath, config.experimentname, date_time)
        checkpoint_dir = os.path.join(savepath, "model_checkpoints")
        log_dir = os.path.join(savepath, "logs")
        os.makedirs(savepath, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        output_config_json_path = os.path.join(savepath, "used_parameters.json")
        with open(output_config_json_path, 'w') as f:
            json.dump(OmegaConf.to_container(config), f)
    else:
        date_time = None
        savepath = None
        checkpoint_dir = None
        log_dir = None

        # Broadcast the folder names from rank 0 to all other ranks
    if dist.is_initialized():
        obj_list = [date_time, savepath, checkpoint_dir, log_dir]
        dist.broadcast_object_list(obj_list, src=0)
        date_time, savepath, checkpoint_dir, log_dir = obj_list


    paths = SimpleNamespace(
        savepath=savepath,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )

    events = SimpleNamespace(
        nEpochs=config.nEpochs,
        # validation_every_N_epochs=config.validation_every_N_epochs,
        special_save_Epochs=config.special_save_Epochs
    )

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # setup tensorboard

    tb_writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None
    if rank == 0:
        tb_writer.add_text("Parameters", str(config))


    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the dataloader

    training_dataset = hydra.utils.instantiate(
        config.dataset_train)
    train_sampler = None
    train_sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    training_dataloader = hydra.utils.instantiate(
        config.dataloader_train,
        dataset=training_dataset,
        sampler=train_sampler,     # IMPORTANT
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    plot_dataset = hydra.utils.instantiate(config.dataset_plot)
    plot_dataloader = hydra.utils.instantiate(
        config.dataloader_plot,
        dataset=plot_dataset,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup lossfunction

    lossfunction = hydra.utils.instantiate(
        config.lossfunction
        )
    lossfunction = lossfunction.cuda(rank)
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the model

    model = hydra.utils.instantiate(config.model)
    model = model.cuda(rank)
    model = torch.compile(model)
    model = DDP(
        model, 
        device_ids=[rank], 
        output_device=rank, 
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        broadcast_buffers=False)
       
    if rank == 0:
        summary(model)

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the optimizer
    optimizer = hydra.utils.instantiate(
        config.optimizer,
        params=model.parameters(),
    )

    scheduler = hydra.utils.instantiate(
        config.scheduler,
        optimizer=optimizer
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # setup the validation metrics

    list_of_metrics, metric_maximize_list = hydra.utils.instantiate(config.validation_metrics)

    metric_coll = MetricCollection(list_of_metrics).cuda(rank)
    tracker = MetricTracker(metric_coll, maximize=metric_maximize_list)

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the trainer

    trainer = hydra.utils.instantiate(
        config.trainroutine,
        model=model,
        training_dataset=training_dataset,
        training_dataloader=training_dataloader,
        plot_dataset=plot_dataset,
        plot_dataloader=plot_dataloader,
        optimizer=optimizer,
        tracker=tracker,
        lossfunction=lossfunction,
        scheduler=scheduler,
        rank=rank,
        world_size=world_size,
        paths=paths,
        events=events,
        tb_writer=tb_writer,
        validation_epochs=config.validation_epochs
    )
                            
    trainer.fit()

    trainer.finalize()
    destroy_process_group()
    
    return None


def set_seed(seed, deterministic, benchmark):
    import random
    import numpy as np
    assert isinstance(seed, int), "Seed must be an integer"
    assert isinstance(deterministic, bool), "Deterministic must be a boolean"
    assert isinstance(benchmark, bool), "Benchmark must be a boolean"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    return None

def runDistributed(config):

    # seeding
    if config.seed is not None:
        set_seed(config.seed,config.deterministic, config.benchmark)

    world_size = len(config.gpus)

    mp.spawn(setup_and_start_training,
             args=(world_size, config),
             nprocs=world_size)

    return None