# torch_parallel
Example of pytorch DistributedDataParallel, including data parallel and model parallel.

# usage
For the scripts under gpt-4o, use:
```bash
python data_parallel.py
python model_parallel.py
```
For the scripts under pytorch, use:
```bash
python model_data_parallel.py

export MASTER_ADDR=localhost
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 elastic_ddp.py
```

# reference
1. ddp: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
2. torch run: https://pytorch.org/docs/stable/elastic/quickstart.html