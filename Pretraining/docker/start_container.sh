docker build -t anonymuser/lianetpretraining \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) .

docker run -it --gpus all --name lianetpretraining --ipc=host \
  -v ~/git/LIANet/Pretraining/src:/home/user/src \
  -v ~/Data/LIANet_data:/home/user/data \
  -v ~/Results/LIANet_results/Pretraining:/home/user/pretraining_results \
anonymuser/lianetpretraining