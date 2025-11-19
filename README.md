🛰️ Location Is All You Need (LIANet)

Official Repository for the Paper:  
📄 *Location Is All You Need: Continuous Spatiotemporal Neural Representations of Earth Observation Data*

This repository contains the implementation of **LIANet**, a neural architecture designed for learning continuous spatiotemporal representations from Earth Observation (EO) data.  
It includes scripts for pretraining and fine-tuning, needed to reproduce the results presented in the paper. The corresponding dataset and labels will be open sourced upon acceptance.

The repository is structured into two main components:

* **Pretraining:** Learn general continuous spatiotemporal representations.  
* **Fine-tuning:** Adapt pretrained weights for specific downstream EO tasks (e.g., landcover classification, building footprint detection).

By default, LIANet expects data and labels to be present in `~/Data/LIANet_data`.  
The pretraining results will be stored in `~/Results/LIANet_results/Pretraining` and fine-tuning results will be stored in `~/Results/LIANet_results/Finetuning`.

---

## 📁 Repository Structure

```text
~git/
└─ LIANet/
   ├─ Finetuning/
   │  ├─ docker/
   │  └─ src/
   │     ├─ configs/                     # YAML configs for fine-tuning runs
   │     ├─ models/                      # Model definitions for downstream tasks
   │     ├─ datasets.py                  # Dataset loaders for fine-tuning tasks
   │     ├─ lr_scheduler.py              # Learning rate scheduling utilities
   │     ├─ metrics.py                   # Evaluation metrics
   │     ├─ settings.py                  # Global constants and paths
   │     ├─ train.py                     # Main training entry point (fine-tuning)
   │     └─ utils.py                     # Helper functions
   │
   ├─ Pretraining/
   │  ├─ docker/
   │  └─ src/
   │     ├─ configs/                     # Pretraining configuration files
   │     ├─ models/                      # LIANet model class
   │     ├─ schedulers/                  # Learning rate schedulers for pretraining
   │     ├─ trainroutines/               # Training loops / routines
   │     ├─ dataset.py                   # Dataset definitions for pretraining
   │     ├─ main.py                      # Main entry script for pretraining
   │     ├─ metrics.py                   # Metrics used during pretraining
   │     ├─ trainer.py                   # Model trainer class
   │     └─ utils.py                     # General utility functions
   │
   ├─ .gitignore
   └─ README.md
```
📦 Data Directory
```text
~/Data/LIANet_data/
├─ DLT.tif                    # Dominant Leaf Type label raster
├─ dw_0.tif                   # Dynamic World label (season index 0)
├─ dw_1.tif                   # Dynamic World label (season index 1)
├─ dw_2.tif                   # Dynamic World label (season index 2)
├─ dw_3.tif                   # Dynamic World label (season index 3)
├─ mbf_binary.tif             # Building footprint binary mask
├─ mbf_density.tif            # Building footprint density
├─ mch.tif                    # Meta Canopy Height (regression label)
├─ s2_seasonidx0.tif          # Sentinel-2 input image (season index 0)
├─ s2_seasonidx1.tif          # Sentinel-2 input image (season index 1)
├─ s2_seasonidx2.tif          # Sentinel-2 input image (season index 2)
└─ s2_seasonidx3.tif          # Sentinel-2 input image (season index 3)
```

And Results Directory
```text
~/Results/
└─ LIANet_results/
   ├─ Pretraining/             # Stores pretraining runs and checkpoints
   └─ Finetuning/              # Stores fine-tuning results
      ├─ dynamic_world/
      └─ ...
```
