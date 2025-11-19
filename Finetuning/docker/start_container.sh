docker build  -t anonymuser/lianetfinetuning .


docker run -it --gpus all --name lianetfinetuning --ipc=host \
  --shm-size 48G \
  --gpus all \
  -p 6632:6632 \
  -v ~/git/LIANet/Finetuning/src:/home/user/src \
  -v ~/Data/LIANet_data:/home/user/data \
  -v ~/Results/LIANet_results/Pretraining:/home/user/pretraining_results \
  -v ~/Results/LIANet_results/Finetuning:/home/user/finetuning_results \
anonymuser/lianetfinetuning