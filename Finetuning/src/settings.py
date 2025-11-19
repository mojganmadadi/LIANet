TOP_DIR = "/home/user/data"
S2_TILES = ["s2_seasonidx0.tif", "s2_seasonidx1.tif", "s2_seasonidx2.tif", "s2_seasonidx3.tif"]

# The dynamic world label's format is "dw_YYYYMMDD_seasonIndex_month.tif"
labels = {"dynamic_world": ["dw_0.tif", "dw_1.tif", "dw_2.tif", "dw_3.tif"],
          "meta_canopy_height": "mch.tif",
          "building_footprints": "mbf_fractional.tif",
          "dominant_leaf_type": "DLT.tif",
          "building_footprints_binary": "mbf_binary.tif" }

# This dictionary includes the paths to the pretrained model checkpoints folders (do not put the .pt file!)
models = {
    "5k_small" : "/home/user/pretraining_results/model_checkpoint/YYYY-MM-DD_HH-MM-SS", # Example path
    "7k_small" : "path/to/7k_small/model_checkpoint",
    "10k_small": "path/to/10k_small/model_checkpoint",
    "5k_large" : "path/to/5k_large/model_checkpoint",
    "7k_large" : "path/to/7k_large/model_checkpoint",
    "10k_large": "path/to/10k_large/model_checkpoint",
    }

# the areas which corresponfs to A0, A+, and A++
area = {
    "5k":5000,
    "7k":7071,
    "10k":10980
    }

num_classes = {
    "dynamic_world": 6,
    "meta_canopy_height": 1,
    "building_footprints": 1,
    "dominant_leaf_type": 3,
    "building_footprints_binary": 2
    }

activation_functions = {
    "dynamic_world": "none",
    "meta_canopy_height": "leakyrelu",
    "building_footprints": "leakyrelu",
    "dominant_leaf_type": "none",
    "building_footprints_binary": "none"
    }