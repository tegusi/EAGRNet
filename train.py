import argparse

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import os
import os.path as osp
from networks.EAGR import EAGRNet
from dataset.datasets import HelenDataSet
import torchvision.transforms as transforms
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess, SingleGPU
from utils.criterion import CriterionAll, CriterionCrossEntropyEdgeParsing_boundary_attention_loss
from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from evaluate import valid
from datetime import datetime
from torch.utils.data.distributed import DistributedSampler
from inplace_abn import InPlaceABN, InPlaceABNSync

TIMESTAMP = "helen"+"{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
import matplotlib.pyplot as plt

start = timeit.default_timer()
  
BATCH_SIZE = 8
DATA_DIRECTORY = './datasets/Helen'
IGNORE_LABEL = 255
INPUT_SIZE = '473,473'
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 11
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './snapshots/resnet101-imagenet.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 10000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'val', 'trainval', 'test'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--test-fre", type=int, default=1,
                        help="Number of classes to predict (including background).")                                            
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers") 
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def adjust_learning_rate_pose(optimizer, epoch):
    decay = 0
    if epoch + 1 >= 230:
        decay = 0.05
    elif epoch + 1 >= 200:
        decay = 0.1
    elif epoch + 1 >= 120:
        decay = 0.25
    elif epoch + 1 >= 90:
        decay = 0.5
    else:
        decay = 1

    lr = args.learning_rate * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003


def main():
    """Create the model and start the training."""

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]
    best_f1 = 0

    torch.cuda.set_device(args.local_rank)

    try:
        world_size = int(os.environ['WORLD_SIZE'])
        distributed = world_size > 1
    except:
        distributed = False
        world_size = 1
    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method='env://')
    rank = 0 if not distributed else dist.get_rank()

    writer = SummaryWriter(osp.join(args.snapshot_dir, TIMESTAMP)) if rank == 0 else None

    cudnn.enabled = True
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
 
    if distributed:
        model = EAGRNet(args.num_classes)
    else:
        model = EAGRNet(args.num_classes, InPlaceABN)
    if args.restore_from is not None:
        model.load_state_dict(torch.load(args.restore_from), True)
    else:
        resnet_params = torch.load(os.path.join(args.snapshot_dir, 'resnet101-imagenet.pth'))
        new_params = model.state_dict().copy()
        for i in resnet_params:
            i_parts = i.split('.')
            # print(i_parts)
            if not i_parts[0] == 'fc':
                new_params['.'.join(i_parts[0:])] = resnet_params[i]
        model.load_state_dict(new_params)
    model.cuda()
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        model = SingleGPU(model)

    criterion = CriterionCrossEntropyEdgeParsing_boundary_attention_loss(loss_weight=[1, 1, 4])
    criterion.cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset = HelenDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    trainloader = data.DataLoader(train_dataset, batch_size=args.batch_size , shuffle=False, num_workers=2,
                                  pin_memory=True, drop_last=True, sampler=train_sampler)
    val_dataset = HelenDataSet(args.data_dir, 'test', crop_size=input_size, transform=transform)
    num_samples = len(val_dataset)
    
    valloader = data.DataLoader(val_dataset, batch_size=args.batch_size ,
                                 shuffle=False, pin_memory=True, drop_last=False)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optimizer.zero_grad()

    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if distributed:
            train_sampler.set_epoch(epoch)
        for i_iter, batch in enumerate(trainloader):
            i_iter += len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, edges, _ = batch
            labels = labels.long().cuda(non_blocking=True)
            edges = edges.long().cuda(non_blocking=True)

            preds = model(images)
            

            loss = criterion(preds, [labels, edges])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss = loss.detach() * labels.shape[0]
                count = labels.new_tensor([labels.shape[0]], dtype=torch.long)
                if dist.is_initialized():
                    dist.all_reduce(count, dist.ReduceOp.SUM)
                    dist.all_reduce(loss, dist.ReduceOp.SUM)
                loss /= count.item()

            if not dist.is_initialized() or dist.get_rank() == 0:
                if i_iter % 50 == 0:
                    writer.add_scalar('learning_rate', lr, i_iter)
                    writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)

                if i_iter % 500 == 0:

                    images_inv = inv_preprocess(images, args.save_num_images)
                    labels_colors = decode_parsing(labels, args.save_num_images, args.num_classes, is_pred=False)
                    edges_colors = decode_parsing(edges, args.save_num_images, 2, is_pred=False)

                    if isinstance(preds, list):
                        preds = preds[0]
                    preds_colors = decode_parsing(preds[0], args.save_num_images, args.num_classes, is_pred=True)
                    pred_edges = decode_parsing(preds[1], args.save_num_images, 2, is_pred=True)

                    img = vutils.make_grid(images_inv, normalize=False, scale_each=True)
                    lab = vutils.make_grid(labels_colors, normalize=False, scale_each=True)
                    pred = vutils.make_grid(preds_colors, normalize=False, scale_each=True)
                    edge = vutils.make_grid(edges_colors, normalize=False, scale_each=True)
                    pred_edge = vutils.make_grid(pred_edges, normalize=False, scale_each=True)


                    writer.add_image('Images/', img, i_iter)
                    writer.add_image('Labels/', lab, i_iter)
                    writer.add_image('Preds/', pred, i_iter)
                    writer.add_image('Edge/', edge, i_iter)
                    writer.add_image('Pred_edge/', pred_edge, i_iter)
    
                print('iter = {} of {} completed, loss = {}'.format(i_iter, total_iters, loss.data.cpu().numpy()))
        if not dist.is_initialized() or dist.get_rank() == 0:
            torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'epoch_' + str(epoch) + '.pth'))

            if epoch % args.test_fre == 0:
                parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples)
                mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'test', True)
                if f1['overall'] > best_f1:
                    torch.save(model.module.state_dict(), osp.join(args.snapshot_dir, TIMESTAMP, 'best.pth'))
                    best_f1 = f1['overall']
                print(mIoU)
                print(f1)
                writer.add_scalars('mIoU', mIoU, epoch)
                writer.add_scalars('f1', f1, epoch)


    end = timeit.default_timer()
    print(end - start, 'seconds')
 

if __name__ == '__main__':
    main()
