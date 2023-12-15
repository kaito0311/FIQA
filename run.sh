export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train_cr_fiqa.py
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 python preprocess_data.py
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 python train_custom_cr_qifa.py

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 python prepare_data.py
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 python train_custom_cr_qifa.py