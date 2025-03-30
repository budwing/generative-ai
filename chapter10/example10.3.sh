torchrun --nnodes=2 --node_rank=0 \
          --master_addr="192.168.1.100" \
          --nproc_per_node=4 train.py