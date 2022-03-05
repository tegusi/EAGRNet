# Edge-aware Graph Representation Learning and Reasoning for Face Parsing

The official repository of *Edge-aware Graph Representation Learning and Reasoning for Face Parsing (ECCV 2020)* and *AGRNet: Adaptive Graph Representation Learning and Reasoning for Face Parsing (TIP 2021)*. 


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
You can download original datasets without alignment:
- **Helen** : [https://www.sifeiliu.net/face-parsing](https://www.sifeiliu.net/face-parsing)
- **LaPa** : [https://github.com/JDAI-CV/lapa-dataset](https://github.com/JDAI-CV/lapa-dataset)
- **CelebAMask-HQ** : [https://github.com/switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)

and put them in `./dataset` folder.
If you need imagenet pretrained resent-101, please download from [baidu drive](https://pan.baidu.com/s/1NoxI_JetjSVa7uqgVSKdPw) or [Google drive](https://drive.google.com/open?id=1rzLU-wK6rEorCNJfwrmIu5hY2wRMyKTK), and put it into snapshot folder. We do not provide the registration code for the moment, and you need to organize input data as follows:

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

We support single-gpu and multi-gpu training. Inplace-abn requires pytorch distributed data parallel. And you can switch between the model between EAGRNet and AGRNet in `train.py`.

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

## Reference

If you consider use our code, please cite our paper:

```
@inproceedings{te2020edge,
  title={Edge-aware Graph Representation Learning and Reasoning for Face Parsing},
  author={Te, Gusi and Liu, Yinglu and Hu, Wei and Shi, Hailin and Mei, Tao},
  booktitle={European Conference on Computer Vision},
  pages={258--274},
  year={2020},
  organization={Springer}
}

@article{te2021agrnet,
  title={Agrnet: Adaptive graph representation learning and reasoning for face parsing},
  author={Te, Gusi and Hu, Wei and Liu, Yinglu and Shi, Hailin and Mei, Tao},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={8236--8250},
  year={2021},
  publisher={IEEE}
}
```

## Acknowledgement

Thanks [@lucia123](https://github.com/lucia123) and her work [A New Dataset and Boundary-Attention Semantic Segmentation for Face Parsing](https://aaai.org/ojs/index.php/AAAI/article/view/6832/6686) in AAAI, 2020.
