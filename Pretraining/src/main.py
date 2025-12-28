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
            config_name="config_fourier.yaml")
def main(config: DictConfig) -> None:
    # must happen before importing anything that imports torch
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpus))


    from trainer import runDistributed 
    runDistributed(config)



if __name__ == "__main__":
    main()