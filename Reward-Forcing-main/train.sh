torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 --rdzv_backend=c10d  \
    --rdzv_endpoint=$MASTER_PORT train.py  --config_path configs/reward_forcing.yaml \
    --logdir logs/reward_forcing \
    --disable-wandb

