docker build -t mmadadi/lianetpretraining \
  --build-arg HOST_UID=$(id -u) \
  --build-arg HOST_GID=$(id -g) .

docker run -it --gpus all --name lianetpretraining --ipc=host \
  -v ~/git/LIANet/Pretraining/src:/home/user/src \
  -v ~/Data/LIANet_data:/home/user/data_local \
  -v /datatank/mojgan.madadikhaljan/LIANet_data:/home/user/data_shared \
  -v ~/Results/LIANet_results/Pretraining:/home/user/results_local \
  -v /datatank/mojgan.madadikhaljan/LIANet_results:/home/user/results_shared \
mmadadi/lianetpretraining