import time
import torch
import os, numpy as np
import os.path as osp
import shutil
import cv2

def generate_edge(label_dir, edge_dir):
    """Generate edges for labels in label_dir and save them to edge_dir
    """
    print('Generating edges from {} to {}'.format(label_dir, edge_dir))
    if os.path.exists(edge_dir):
        shutil.rmtree(edge_dir)
    os.makedirs(edge_dir)
    ll = os.listdir(label_dir)
    for filename in ll:
        print(filename)
        label = cv2.imread(osp.join(label_dir, filename), cv2.IMREAD_GRAYSCALE)
        edge = np.zeros_like(label)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                flag = 1
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x = i + dx
                    y = j + dy
                    if 0 <= x < label.shape[0] and 0 <= y < label.shape[1]:
                        if label[i,j] != label[x,y]:
                            edge[i,j] = 255
        cv2.imwrite(osp.join(edge_dir, filename), edge)

generate_edge('dataset/helen/labels/', 'dataset/helen/edges/')