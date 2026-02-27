docker build  -t mmadadi/lianetfinetuning .


docker run -it --gpus all --name lianetfinetuning --ipc=host \
  --shm-size 48G \
  --gpus all \
  -p 6612:6612 \
  -v ~/git/LIANet/Finetuning/src:/home/user/src \
  -v ~/Data/LIANet_data:/home/user/data_local \
  -v /datatank/mojgan.madadikhaljan/LIANet_data:/home/user/data_shared  \
  -v ~/Results/LIANet_results/Finetuning:/home/user/results_local \
  -v /datatank/mojgan.madadikhaljan/LIANet_results/:/home/user/results_shared \
mmadadi/lianetfinetuning


