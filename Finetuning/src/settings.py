TOP_DIR = {"dynamic_world": "/home/user/data_shared/T32UPU",
          "meta_canopy_height": "/home/user/data_shared/T32UPU",
          "building_footprints": "/home/user/data_shared/T32UPU",
          "dominant_leaf_type": "/home/user/data_shared/T32UPU",
          "building_footprints_binary": "/home/user/data_shared/T32UPU",
          "PASTIS_T32ULU":"/home/user/data_shared",
          "PASTIS_T31TFM":"/home/user/data_shared",
          "BurnScars":"/home/user/data_shared"}

fourseason_s2_list   = ['20240619T102031.tif', '20220715T101559.tif', '20250219T101959.tif', '20220615T101559.tif', '20251123T102401.tif', '20240225T101919.tif', '20231003T101841.tif', '20250619T101559.tif', '20240828T102021.tif', '20220630T102041.tif', '20230531T101559.tif', '20250808T101559.tif', '20251002T101851.tif', '20250430T101559.tif', '20220824T101559.tif', '20230625T101601.tif', '20240813T101559.tif', '20250510T101559.tif', '20220804T101559.tif', '20230819T101609.tif', '20240907T102021.tif', '20230210T102049.tif', '20240729T102021.tif', '20231207T102319.tif', '20250818T101559.tif', '20220814T101559.tif', '20230715T101601.tif', '20231013T101951.tif', '20241221T102339.tif', '20250405T102041.tif', '20230908T101559.tif', '20230928T101719.tif', '20221018T102031.tif', '20240709T102031.tif', '20240205T102129.tif', '20220809T102041.tif', '20240624T101559.tif', '20250813T102041.tif', '20250609T101559.tif', '20250318T101751.tif', '20230824T101601.tif', '20250407T101701.tif', '20220725T101559.tif']


s2_tiles = {"dynamic_world": fourseason_s2_list,
          "meta_canopy_height": fourseason_s2_list,
          "building_footprints": fourseason_s2_list,
          "dominant_leaf_type": fourseason_s2_list,
          "building_footprints_binary": fourseason_s2_list,
          "PASTIS_T32ULU": "T32ULU", # or "T31TFM"
          "PASTIS_T31TFM": "T31TFM", # or "T31TFM"
          "BurnScars": "T11SMT"}
# The dynamic world label's format is "dw_YYYYMMDD_seasonIndex_month.tif"
labels = {"dynamic_world": ["dw_0.tif", "dw_1.tif", "dw_2.tif", "dw_3.tif"],
          "meta_canopy_height": "mch.tif",
          "building_footprints": "mbf_fractional.tif",
          "dominant_leaf_type": "DLT.tif",
          "building_footprints_binary": 'BF_20250818T101559_20250818T101559_2.5m.tif',
          "PASTIS_T32ULU": "masks/PASTIS",
          "PASTIS_T31TFM": "masks/PASTIS",
          "BurnScars": "masks/BurnScars"}

# This dictionary includes the paths to the pretrained model checkpoints folders (do not put the .pt file!)
models = {
    "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_EU/2026-01-11_12-03-58", # Example path
    # "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_USA/2026-01-15_12-33-30", # Example path
    "7k_small" : "path/to/7k_small/model_checkpoint",
    "10k_small": "path/to/10k_small/model_checkpoint",
    "5k_large" : "path/to/5k_large/model_checkpoint",
    "7k_large" : "path/to/7k_large/model_checkpoint",
    "10k_large": "path/to/10k_large/model_checkpoint",
    "full_tile_modified_PASTIS_T31TFM" :"/home/user/results_shared/fourier_learned_2tile_alltimes_france_largermodel_nocloud/2026-02-19_15-31-25",
    "full_tile_modified_PASTIS_T32ULU" :"/home/user/results_shared/fourier_learned_2tile_alltimes_france_largermodel_nocloud/2026-02-12_10-12-55_T32ULU",
    "full_tile_modified_BurnScars" :"/home/user/results_shared/fourier_learned_HLS_USA/2026-02-22_13-57-00",
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
    "building_footprints_binary": 2,
    "PASTIS_T32ULU": 20,
    "PASTIS_T31TFM": 20,
    "BurnScars": 2
    }

activation_functions = {
    "dynamic_world": "none",
    "meta_canopy_height": "leakyrelu",
    "building_footprints": "leakyrelu",
    "dominant_leaf_type": "none",
    "building_footprints_binary": "none",
    "PASTIS_T32ULU": "none",
    "PASTIS_T31TFM": "none",
    "BurnScars": "none"
    }