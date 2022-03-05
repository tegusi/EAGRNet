import os
import numpy as np
import random
import torch
import cv2
import json
from torch.utils import data
import torch.distributed as dist
from utils.transforms import get_affine_transform
import os.path as osp
import math
import face_alignment

def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

class HelenDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.transform = transform
        self.dataset = dataset

        self.file_list_name = osp.join(root, dataset + '_list.txt')
        # self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device='cuda:1')
        # self.im_list = []
        # i = 0
        # for line in open(self.file_list_name).readlines():
        #     i += 1
        #     im_path = os.path.join(self.root, 'images', line.split()[0][7:-4] + '.jpg')
        #     im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        #     outlier = self.cal_angle(im)
        #     print(outlier)
        #     if dataset == 'train' and outlier:
        #         self.im_list.append(line.split()[0][7:-4])
        #     if dataset != 'train' and not outlier:
        #         self.im_list.append(line.split()[0][7:-4])
        self.im_list = [line.split()[0][7:-4] for line in open(self.file_list_name).readlines()]
        self.number_samples = len(self.im_list)
        print(self.number_samples)

    def cal_angle(self, img):
        imagePoints = self.fa.get_landmarks(img)
        if(imagePoints is not None):
            imagePoints = imagePoints[0]
        # Compute the Mean-Centered-Scaled Points
        mean = np.mean(imagePoints, axis=0) # <- This is the unscaled mean
        scaled = (imagePoints / np.linalg.norm(imagePoints[42] - imagePoints[39])) * 0.06 # Set the inner eye distance to 60cm (just because)
        centered = scaled - np.mean(scaled, axis=0) # <- This is the scaled mean
        # Construct a "rotation" matrix (strong simplification, might have shearing)
        rotationMatrix = np.empty((3,3))
        rotationMatrix[0,:] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0])
        rotationMatrix[1,:] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
        rotationMatrix[2,:] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
        eulerAngles=rotationMatrixToEulerAngles(rotationMatrix)
        pitch, yaw, roll = [math.radians(x)*100.0 for x in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        # rotate_degree=[ str(int(pitch)), str(int(roll)),str(int(yaw))]
        return math.floor(pitch) > 40 or math.floor(pitch) < 0
        # return (math.floor(pitch),math.floor(roll),math.floor(yaw))

    def __len__(self):
        return self.number_samples 

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, 'images', im_name + '.jpg')
        edge_path = os.path.join(self.root, 'edges', im_name + '.png')
        parsing_anno_path = os.path.join(self.root, 'labels', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test': 
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            if self.dataset in 'train':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset not in 'train':
            return input, edge, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, edge, meta

class LapaDataset(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.transform = transform
        self.dataset = dataset

        self.im_list = [line[:-5] for line in open(osp.join(root, dataset + '_id.txt')).readlines()]
        #self.im_list = self.im_list[:40]
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset, 'images', im_name + '.jpg')
        edge_path = os.path.join(self.root, self.dataset, 'edges', im_name + '.png')
        parsing_anno_path = os.path.join(self.root, self.dataset, 'labels', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test': 
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset in 'train':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                # if random.random() <= self.flip_prob:
                #     im = im[:, ::-1, :]
                #     parsing_anno = parsing_anno[:, ::-1]

                #     center[0] = im.shape[1] - center[0] - 1
                #     right_idx = [5, 7, 9]
                #     left_idx = [4, 6, 8]
                #     for i in range(0, len(left_idx)):
                #         right_pos = np.where(parsing_anno == right_idx[i])
                #         left_pos = np.where(parsing_anno == left_idx[i])
                #         parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                #         parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset not in 'train':
            return input, edge, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, edge, meta

class CelebAMaskHQDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[4, 5], [6, 7]]
        self.transform = transform
        self.dataset = dataset

        list_path = os.path.join(self.root, self.dataset + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        #self.im_list = self.im_list[:40]
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset, 'images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset, 'labels', im_name + '.png')
        edge_path = os.path.join(self.root, self.dataset, 'edges', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test': 
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset in 'train':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                # if random.random() <= self.flip_prob:
                #     im = im[:, ::-1, :]
                #     parsing_anno = parsing_anno[:, ::-1]

                #     center[0] = im.shape[1] - center[0] - 1
                #     right_idx = [5, 7, 9]
                #     left_idx = [4, 6, 8]
                #     for i in range(0, len(left_idx)):
                #         right_pos = np.where(parsing_anno == right_idx[i])
                #         left_pos = np.where(parsing_anno == left_idx[i])
                #         parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                #         parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset not in 'train':
            return input, edge, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, edge, meta

class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset

        list_path = os.path.join(self.root, self.dataset + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset, 'images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset, 'labels', im_name + '.png')
        edge_path = os.path.join(self.root, self.dataset, 'edges', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test': 
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset in 'train':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                # if random.random() <= self.flip_prob:
                #     im = im[:, ::-1, :]
                #     parsing_anno = parsing_anno[:, ::-1]

                #     center[0] = im.shape[1] - center[0] - 1
                #     right_idx = [5, 7, 9]
                #     left_idx = [4, 6, 8]
                #     for i in range(0, len(left_idx)):
                #         right_pos = np.where(parsing_anno == right_idx[i])
                #         left_pos = np.where(parsing_anno == left_idx[i])
                #         parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                #         parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge = cv2.warpAffine(
            edge,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        edge[np.where(edge != 0)] = 1

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset not in 'train':
            return input, edge, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, edge, meta