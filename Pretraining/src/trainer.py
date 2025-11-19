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


def setup_and_start_training(config: DictConfig):
        
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # save the config and extract relevant parameters

    rank = 0
    world_size = 1

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savepath = os.path.join(config.outputpath, config.experimentname, date_time)
    checkpoint_dir = os.path.join(savepath, "model_checkpoints")
    log_dir = os.path.join(savepath, "logs")
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    output_config_json_path = os.path.join(savepath, "used_parameters.json")
    with open(output_config_json_path, 'w') as f:
        json.dump(OmegaConf.to_container(config), f)

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

    tb_writer = SummaryWriter(log_dir=log_dir)
    tb_writer.add_text("Parameters",str(config))

    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the dataloader

    training_dataset = hydra.utils.instantiate(
        config.dataset_train)

    training_dataloader = hydra.utils.instantiate(
        config.dataloader_train,
        dataset=training_dataset,
    )    

    plot_dataset = hydra.utils.instantiate(
        config.dataset_plot)

    plot_dataloader = hydra.utils.instantiate(
        config.dataloader_plot,
        dataset=plot_dataset,
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup lossfunction

    lossfunction = hydra.utils.instantiate(
        config.lossfunction
        )
    lossfunction = lossfunction.cuda()
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++
    # Setup the model

    model = hydra.utils.instantiate(
                        config.model)

    model = model.cuda()             
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

    metric_coll = MetricCollection(list_of_metrics).cuda()
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

    return None