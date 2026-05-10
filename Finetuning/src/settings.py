TOP_DIR = {"dynamic_world": "/home/user/data_shared/T32UPU",
          "meta_canopy_height": "/home/user/data_shared/T32UPU",
          "BFPBinary_joint_T31TFM": "/home/user/data_shared",
          "BFPBinary_joint_T32ULU": "/home/user/data_shared",
          "BFPBinary_local_T31TFM": "/home/user/data_shared",
          "BFPBinary_local_T32ULU": "/home/user/data_shared",          
          "dominant_leaf_type": "/home/user/data_shared/T32UPU",
          "BFPDensity_joint_T31TFM": "/home/user/data_shared",
          "BFPDensity_joint_T32ULU": "/home/user/data_shared",
          "BFPDensity_local_T31TFM": "/home/user/data_shared",
          "BFPDensity_local_T32ULU": "/home/user/data_shared",  
          "PASTIS_joint_T32ULU":"/home/user/data_shared",
          "PASTIS_joint_T31TFM":"/home/user/data_shared",
          "PASTIS_joint_T30UXV":"/home/user/data_shared",
          "PASTIS_joint_T31TFJ":"/home/user/data_shared",
          "PASTIS_local_T32ULU":"/home/user/data_shared",
          "PASTIS_local_T31TFM":"/home/user/data_shared",
          "PASTIS_local_T30UXV":"/home/user/data_shared",
          "PASTIS_local_T31TFJ":"/home/user/data_shared",
          "BurnScars_joint_T11SMT":"/home/user/data_shared",
          "BurnScars_joint_T16REV":"/home/user/data_shared",
          "BurnScars_local_T11SMT":"/home/user/data_shared",
          "BurnScars_local_T16REV":"/home/user/data_shared"}

fourseason_s2_list   = ['20240619T102031.tif', '20220715T101559.tif', '20250219T101959.tif', '20220615T101559.tif', '20251123T102401.tif', '20240225T101919.tif', '20231003T101841.tif', '20250619T101559.tif', '20240828T102021.tif', '20220630T102041.tif', '20230531T101559.tif', '20250808T101559.tif', '20251002T101851.tif', '20250430T101559.tif', '20220824T101559.tif', '20230625T101601.tif', '20240813T101559.tif', '20250510T101559.tif', '20220804T101559.tif', '20230819T101609.tif', '20240907T102021.tif', '20230210T102049.tif', '20240729T102021.tif', '20231207T102319.tif', '20250818T101559.tif', '20220814T101559.tif', '20230715T101601.tif', '20231013T101951.tif', '20241221T102339.tif', '20250405T102041.tif', '20230908T101559.tif', '20230928T101719.tif', '20221018T102031.tif', '20240709T102031.tif', '20240205T102129.tif', '20220809T102041.tif', '20240624T101559.tif', '20250813T102041.tif', '20250609T101559.tif', '20250318T101751.tif', '20230824T101601.tif', '20250407T101701.tif', '20220725T101559.tif']


s2_tiles = {"dynamic_world": fourseason_s2_list,
          "meta_canopy_height": fourseason_s2_list,
          "BFPBinary_joint_T31TFM": "T31TFM",
          "BFPBinary_joint_T32ULU": "T32ULU",
          "BFPBinary_local_T31TFM": "T31TFM",
          "BFPBinary_local_T32ULU": "T32ULU",
          "dominant_leaf_type": fourseason_s2_list,
          "BFPDensity_joint_T31TFM": "T31TFM",
          "BFPDensity_joint_T32ULU": "T32ULU",
          "BFPDensity_local_T31TFM": "T31TFM",
          "BFPDensity_local_T32ULU": "T32ULU",
          "building_footprints_binary": fourseason_s2_list,
          "PASTIS_joint_T32ULU": "T32ULU", # or "T31TFM"
          "PASTIS_joint_T31TFM": "T31TFM", # or "T31TFM"
          "PASTIS_joint_T30UXV": "T30UXV",
          "PASTIS_joint_T31TFJ": "T31TFJ",
          "PASTIS_local_T32ULU": "T32ULU", # or "T31TFM"
          "PASTIS_local_T31TFM": "T31TFM", # or "T31TFM"
          "PASTIS_local_T30UXV": "T30UXV",
          "PASTIS_local_T31TFJ": "T31TFJ",
          "BurnScars_joint_T11SMT": "T11SMT",
          "BurnScars_joint_T16REV": "T16REV",
          "BurnScars_local_T11SMT": "T11SMT",
          "BurnScars_local_T16REV": "T16REV"}
# The dynamic world label's format is "dw_YYYYMMDD_seasonIndex_month.tif"
labels = {"dynamic_world": ["dw_0.tif", "dw_1.tif", "dw_2.tif", "dw_3.tif"],
          "meta_canopy_height": "mch.tif",
          "BFPBinary_joint_T31TFM": "T31TFM_20180920T104019_microsoft_buildings_2p5m.tif",
          "BFPBinary_joint_T32ULU": "T32ULU_20180917T103019_microsoft_buildings_2p5m.tif",
          "BFPBinary_local_T31TFM": "T31TFM_20180920T104019_microsoft_buildings_2p5m.tif",
          "BFPBinary_local_T32ULU": "T32ULU_20180917T103019_microsoft_buildings_2p5m.tif",
          "dominant_leaf_type": "DLT.tif",
          "BFPDensity_joint_T31TFM": "T31TFM_20180920T104019_microsoft_buildings_2p5m.tif",
          "BFPDensity_joint_T32ULU": "T32ULU_20180917T103019_microsoft_buildings_2p5m.tif",
          "BFPDensity_local_T31TFM": "T31TFM_20180920T104019_microsoft_buildings_2p5m.tif",
          "BFPDensity_local_T32ULU": "T32ULU_20180917T103019_microsoft_buildings_2p5m.tif",
          "PASTIS_joint_T32ULU": "masks/PASTIS",
          "PASTIS_joint_T31TFM": "masks/PASTIS",
          "PASTIS_joint_T30UXV": "masks/PASTIS",
          "PASTIS_joint_T31TFJ": "masks/PASTIS",
          "PASTIS_local_T32ULU": "masks/PASTIS",
          "PASTIS_local_T31TFM": "masks/PASTIS",
          "PASTIS_local_T30UXV": "masks/PASTIS",
          "PASTIS_local_T31TFJ": "masks/PASTIS",
          "BurnScars_joint_T11SMT": "masks/BurnScars",
          "BurnScars_joint_T16REV": "masks/BurnScars",
          "BurnScars_local_T11SMT": "masks/BurnScars",
          "BurnScars_local_T16REV": "masks/BurnScars"}

# This dictionary includes the paths to the pretrained model checkpoints folders (do not put the .pt file!)
models = {
    "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_EU/2026-01-11_12-03-58", # Example path
    # "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_USA/2026-01-15_12-33-30", # Example path
    "7k_small" : "path/to/7k_small/model_checkpoint",
    "10k_small": "path/to/10k_small/model_checkpoint",
    "5k_large" : "path/to/5k_large/model_checkpoint",
    "7k_large" : "path/to/7k_large/model_checkpoint",
    "10k_large": "path/to/10k_large/model_checkpoint",
    "PASTIS_joint_T32ULU" :"/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "PASTIS_joint_T31TFM" :"/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "PASTIS_joint_T30UXV" :"/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "PASTIS_joint_T31TFJ" :"/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "PASTIS_local_T31TFM" :"/home/user/results_shared/fourier_learned_T31TFM/2026-04-08_00-13-56",
    "PASTIS_local_T32ULU" :"/home/user/results_shared/fourier_learned_T32ULU/2026-04-04_09-21-54",
    "PASTIS_local_T30UXV" :"/home/user/results_shared/fourier_learned_T30UXV/2026-04-06_04-50-55",
    "PASTIS_local_T31TFJ" :"/home/user/results_shared/fourier_learned_T31TFJ/2026-04-02_13-40-05",
    "BurnScars_joint_T11SMT" :"/home/user/results_shared/fourier_learned_HLS_2regions/2026-04-22_15-34-35",
    "BurnScars_joint_T16REV" :"/home/user/results_shared/fourier_learned_HLS_2regions/2026-04-22_15-34-35",
    "BurnScars_local_T11SMT" :"/home/user/results_shared/fourier_learned_HLS_T11SMT/2026-04-30_20-30-35",
    "BurnScars_local_T16REV" :"/home/user/results_shared/fourier_learned_HLS_T16REV/2026-04-28_09-13-10",
    "BFPBinary_joint_T31TFM": "/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "BFPBinary_joint_T32ULU": "/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "BFPBinary_local_T31TFM": "/home/user/results_shared/fourier_learned_T31TFM/2026-04-08_00-13-56",
    "BFPBinary_local_T32ULU": "/home/user/results_shared/fourier_learned_T32ULU/2026-04-04_09-21-54",
    "BFPDensity_joint_T31TFM": "/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "BFPDensity_joint_T32ULU": "/home/user/results_shared/fourier_learned_4regions/2026-03-18_19-06-01",
    "BFPDensity_local_T31TFM": "/home/user/results_shared/fourier_learned_T31TFM/2026-04-08_00-13-56",
    "BFPDensity_local_T32ULU": "/home/user/results_shared/fourier_learned_T32ULU/2026-04-04_09-21-54",
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
    "PASTIS_joint_T32ULU": 19, # the void class (=19) is mapped to 255 and exlduded from training a
    "PASTIS_joint_T31TFM": 19,
    "PASTIS_joint_T30UXV": 19,
    "PASTIS_joint_T31TFJ": 19,
    "PASTIS_local_T32ULU": 19,
    "PASTIS_local_T31TFM": 19,
    "PASTIS_local_T30UXV": 19,
    "PASTIS_local_T31TFJ": 19,
    "BurnScars_joint_T11SMT": 2,
    "BurnScars_joint_T16REV": 2,
    "BurnScars_local_T11SMT": 2,
    "BurnScars_local_T16REV": 2,
    "BFPBinary_joint_T31TFM": 2,
    "BFPBinary_joint_T32ULU": 2,
    "BFPBinary_local_T31TFM": 2,
    "BFPBinary_local_T32ULU": 2,
    "BFPDensity_joint_T31TFM": 1, # Regression
    "BFPDensity_joint_T32ULU": 1, # Regression
    "BFPDensity_local_T31TFM": 1, # Regression
    "BFPDensity_local_T32ULU": 1, # Regression

    }

activation_functions = {
    "dynamic_world": "none",
    "meta_canopy_height": "leakyrelu",
    "dominant_leaf_type": "none",
    "PASTIS_joint_T32ULU": "none",
    "PASTIS_joint_T31TFM": "none",
    "PASTIS_joint_T30UXV": "none",
    "PASTIS_joint_T31TFJ": "none",
    "PASTIS_local_T32ULU": "none",
    "PASTIS_local_T31TFM": "none",
    "PASTIS_local_T30UXV": "none",
    "PASTIS_local_T31TFJ": "none",
    "BurnScars_joint_T11SMT": "none",
    "BurnScars_joint_T16REV": "none",
    "BurnScars_local_T11SMT": "none",
    "BurnScars_local_T16REV": "none",
    "BFPBinary_joint_T31TFM": "none",
    "BFPBinary_joint_T32ULU": "none",
    "BFPBinary_local_T31TFM": "none",
    "BFPBinary_local_T32ULU": "none",
    "BFPDensity_joint_T31TFM": "leakyrelu", # Regression
    "BFPDensity_joint_T32ULU": "leakyrelu", # Regression
    "BFPDensity_local_T31TFM": "leakyrelu", # Regression
    "BFPDensity_local_T32ULU": "leakyrelu", # Regression
    }