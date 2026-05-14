python train.py --config-name "BF_clas_UNet" task=BFPBinary_local_T32ULU batchsize=64 learningrate=1e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_UNet" task=BFPBinary_local_T32ULU batchsize=32 learningrate=3e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_UNet" task=BFPBinary_local_T32ULU batchsize=8 learningrate=1e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_UNet" task=BFPBinary_local_T32ULU batchsize=16 learningrate=3e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_UNet" task=BFPBinary_local_T32ULU batchsize=128 learningrate=1e-6 seed=111 gpu_id=1
#
python train.py --config-name "BF_clas_microUNet" task=BFPBinary_local_T32ULU batchsize=64 learningrate=1e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_microUNet" task=BFPBinary_local_T32ULU batchsize=32 learningrate=3e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_microUNet" task=BFPBinary_local_T32ULU batchsize=8 learningrate=1e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_microUNet" task=BFPBinary_local_T32ULU batchsize=16 learningrate=3e-6 seed=111 gpu_id=1
python train.py --config-name "BF_clas_microUNet" task=BFPBinary_local_T32ULU batchsize=128 learningrate=1e-6 seed=111 gpu_id=1