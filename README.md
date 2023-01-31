# M2GNN
This is our Pytorch implementation for the paper: "M2GNN: Metapath and Multi-interest Aggregated Graph Neural Network for Tag-based Cross-domain Recommendation"

The proposed model is called M2GNN.
## Dataset
DPBJ dataset will be released later because it is being deencrypted.

Amazon dataset has been released.

## Train
Training M2GNN on DPBJ dataset needs at least 4 A100 devices, the start-up command is
~~~
python -m torch.distributed.launch --nproc_per_node 4 main_M2GNN.py --max_K 6 --gamma 7
~~~

Training M2GNN on Amazon dataset needs at least 2 V100 devices, the start-up command is
~~~
python -m torch.distributed.launch --nproc_per_node 2 main_M2GNN_amazon.py --max_K 6 --gamma 7
~~~

## Training log
You can also see the training log to check the convergence process of the model.
