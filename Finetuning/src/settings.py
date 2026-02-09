TOP_DIR = "/home/user/data_shared/T32UPU"
# S2_TILES = ['20221108T163511.tif', '20250814T162921.tif', '20251122T163621.tif', '20221009T163211.tif', '20220914T162839.tif', '20240918T162941.tif', '20240829T162901.tif', '20240211T163421.tif', '20231118T163549.tif', '20250518T163701.tif', '20250918T162839.tif', '20220107T163649.tif', '20240226T163139.tif', '20220721T162851.tif', '20230517T162831.tif', '20221029T163421.tif', '20221203T163649.tif', '20220507T162829.tif', '20250928T162949.tif', '20240406T162829.tif', '20230914T162911.tif', '20221004T163129.tif', '20240824T162829.tif', '20230412T162839.tif', '20220427T162829.tif', '20230924T163021.tif', '20241008T163201.tif', '20220621T162911.tif', '20250908T162829.tif', '20220313T163051.tif', '20250610T162829.tif', '20240725T162839.tif', '20221103T163439.tif', '20250915T163701.tif', '20231123T163611.tif', '20250225T163301.tif', '20241003T163029.tif', '20230711T162839.tif', '20231113T163531.tif', '20221014T163229.tif', '20241018T163311.tif', '20221019T163311.tif', '20230211T163419.tif']
S2_TILES = ['20240619T102031.tif', '20220715T101559.tif', '20250219T101959.tif', '20220615T101559.tif', '20251123T102401.tif', '20240225T101919.tif', '20231003T101841.tif', '20250619T101559.tif', '20240828T102021.tif', '20220630T102041.tif', '20230531T101559.tif', '20250808T101559.tif', '20251002T101851.tif', '20250430T101559.tif', '20220824T101559.tif', '20230625T101601.tif', '20240813T101559.tif', '20250510T101559.tif', '20220804T101559.tif', '20230819T101609.tif', '20240907T102021.tif', '20230210T102049.tif', '20240729T102021.tif', '20231207T102319.tif', '20250818T101559.tif', '20220814T101559.tif', '20230715T101601.tif', '20231013T101951.tif', '20241221T102339.tif', '20250405T102041.tif', '20230908T101559.tif', '20230928T101719.tif', '20221018T102031.tif', '20240709T102031.tif', '20240205T102129.tif', '20220809T102041.tif', '20240624T101559.tif', '20250813T102041.tif', '20250609T101559.tif', '20250318T101751.tif', '20230824T101601.tif', '20250407T101701.tif', '20220725T101559.tif']
# The dynamic world label's format is "dw_YYYYMMDD_seasonIndex_month.tif"
labels = {"dynamic_world": ["dw_0.tif", "dw_1.tif", "dw_2.tif", "dw_3.tif"],
          "meta_canopy_height": "mch.tif",
          "building_footprints": "mbf_fractional.tif",
          "dominant_leaf_type": "DLT.tif",
          "building_footprints_binary": 'BF_20250818T101559_20250818T101559_2.5m.tif'}
        #   "building_footprints_binary": "BF_20250928T162949_20250928T162949_2.5m.tif" }

# This dictionary includes the paths to the pretrained model checkpoints folders (do not put the .pt file!)
models = {
    "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_EU/2026-01-11_12-03-58", # Example path
    # "5k_small" : "/home/user/results_shared/fourier_learned_2tile_alltimes_USA/2026-01-15_12-33-30", # Example path
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