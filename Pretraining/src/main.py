import hydra
from omegaconf import DictConfig

import sys
import os

# Add project root (/home/user/src) to sys.path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


@hydra.main(version_base = None,
            config_path="./configs/pretrain",
            config_name="config.yaml")
def main(config : DictConfig) -> None:

    # we first have to set visible
    # devices before importing any torch libs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config.gpu_idx])

    from trainer import setup_and_start_training 
    setup_and_start_training(config)



if __name__ == "__main__":
    main()