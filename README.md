# Edge-aware Graph Representation Learning and Reasoning for Face Parsing

The official repository of *Edge-aware Graph Representation Learning and Reasoning for Face Parsing (ECCV 2020)*. 


## Installation

Our model is based on Pytorch 1.4.0 with Python 3.6.8. Also, we use [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn). First, you need to clone and compile inplace_abn.

```sh
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
cd scripts
pip install -r requirements.txt
```

## Data

You can download the Helen dataset on [https://www.sifeiliu.net/face-parsing](https://www.sifeiliu.net/face-parsing) and imagenet pretrained resent-101 from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into snapshot folder. We do not provide the registration code for the moment, and you need to organize input data as follows:

```
dataset/
    images/
    labels/
    edges/
    train_list.txt
    test_list.txt
        each line: 'images/100032540_1.jpg labels/100032540_1.png'
```

Besides, we provide the edge genearation code in the *generate_edge.py*.

## Usage

We support single-gpu and multi-gpu training. Inplace-abn requires pytorch distributed data parallel.

Single gpu training
```
python train.py --data-dir ./dataset/Helen/ --random-mirror --random-scale --gpu 0 --learning-rate 1e-3 --weight-decay 5e-4 --batch-size 7 --input-size 473,473 --snapshot-dir ./snapshots/ --dataset train --num-classes 11 --epochs 200
```

Distributed(multi-gpu) training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --data-dir ./dataset/Helen/ --random-mirror --random-scale --gpu 0,1,2,3 --learning-rate 1e-3 --weight-decay 5e-4 --batch-size 7 --input-size 473,473 --snapshot-dir ./snapshots/ --dataset train --num-classes 11 --epochs 99
```

Validation
```
python evaluate.py --data-dir ./dataset/Helen/ --restore-from ./snapshots/helen/best.pth --gpu 0 --batch-size 7 --input-size 473,473 --dataset test --num-classes 11
```
