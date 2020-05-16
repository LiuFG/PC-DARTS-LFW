import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkFace as Network
from data import create_loader, ImageList, FastCollateMixup, resolve_data_config
from mfcode.lfw_eval import validate as lfw_eval
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("training face")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.4, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=400, help='report frequency')
parser.add_argument('--epochs', type=int, default=80, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.3, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--save', type=str, default='./res/full_train/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.2, help='label smoothing')
parser.add_argument('--use_dropout', action='store_true', default=True, help='no')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--root_path', type=str, default='/home/liufg/my_work/data_augment/face/webface_crop/train',
                    metavar='H', help='Dir Head')
# parser.add_argument('--load_path', type=str, default='./res/full_train/eval-try-20200503-120751/model_best.pth.tar')
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--trainFile', type=str, default='trainlist_full.txt', metavar='TRF', help='training file name')
# parser.add_argument('--testFile', type=str, default='test.txt', metavar='TEF',
#                     help='test file name')
parser.add_argument('--val_list', default='/home/liufg/my_work/data_augment/face/LFW/pairs.txt', type=str)
parser.add_argument('--val_folder', default='/home/liufg/my_work/data_augment/face/LFW/aligned', type=str)
parser.add_argument('--val_imlist', default='/home/liufg/my_work/data_augment/face/LFW/image_list.txt', type=str)
parser.add_argument('--prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')
# Augmentation parameters
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')  # change: to 0, original: 0.4
parser.add_argument('--reprob', type=float, default=0.9, metavar='PCT',
                    help='Random erase prob (default: 0.)')  # 0 means switch off ?? to check
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')   # 0.2 to switch on
parser.add_argument('--mixup-off-epoch', default=60, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 10576

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype, args.use_dropout)

    if num_gpus > 1:
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.load_path is not None:
        pretrained_dict = torch.load(os.path.expanduser(args.load_path),
                                     map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()  # 获取模型的参数字典
        model_dict.update(pretrained_dict['state_dict'])  # pretrained_dict与model_dict相同key的value的维度必须相同
        model.load_state_dict(model_dict)  # 更新模型权重
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    dataset_train = ImageList(root=args.root_path, fileList=args.root_path + '/' + args.trainFile)
    collate_fn = None
    if args.prefetcher and args.mixup > 0:
      collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)
    num_train = len(dataset_train)
    train_queue = create_loader(
      dataset_train,  # here call the transform implicitly
      input_size=144,
      batch_size=args.batch_size,
      is_training=True,
      use_prefetcher=args.prefetcher,
      rand_erase_prob=args.reprob,
      rand_erase_mode=args.remode,
      color_jitter=args.color_jitter,
      interpolation='random',  # FIXME cleanly resolve this? data_config['interpolation'],
      # mean=data_config['mean'],
      # std=data_config['std'],
      num_workers=args.workers,
      collate_fn=collate_fn,
      auto_augment=args.aa,
      use_aug=True
    )

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_LFWACC = 0
    lr = args.learning_rate
    writer = SummaryWriter(os.path.join(args.save, 'tensorboard'), comment='pcdarts-for-LFW')
    for epoch in range(args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            print('Wrong lr type, exit')
            sys.exit(1)
        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.load_path is None: # and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / 5.0
                current_lr = lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr)
        if num_gpus > 1:
            model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        else:
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer)
        logging.info('Train_acc: %f', train_acc)
        writer.add_scalar('lr', current_lr, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_loss', train_obj, epoch)

        if (epoch >= 50 and (epoch+1) % 2 == 0) or args.load_path is not None:
            # valid_acc, valid_obj = infer(valid_queue, model, criterion)
            # logging.info('valid_acc %f', valid_acc)
            LFWACC, std, thd = lfw_eval(args, model)
            logging.info('lfw_eval LFW_ACC:%f LFW_std:%f LFW_thd:%f', LFWACC, std, thd)
            writer.add_scalar('LFW_ACC', LFWACC, epoch)
            writer.add_scalar('LFW_std', std, epoch)
            writer.add_scalar('LFW_thd', thd, epoch)
            is_best = False
            if LFWACC >= best_LFWACC:
                best_LFWACC = LFWACC
                is_best = True
                logging.info('lfw_eval BEST_LFW_ACC:%f', best_LFWACC)
            utils.save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': model.state_dict(),
              'best_LFWACC': best_LFWACC,
              'optimizer': optimizer.state_dict(),
            }, is_best, args.save)
    writer.close()

def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs - epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    batch_time = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f Duration: %ds BTime: %.3fs',
                         step, objs.avg, top1.avg, top5.avg, duration, batch_time.avg)
            # writer.add_scalar('train_acc', train_acc, epoch*self.num_batch + step)
            # writer.add_scalar('train_loss', train_obj, epoch*self.num_batch + step)
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            end_time = time.time()
            if step == 0:
                duration = 0
                start_time = time.time()
            else:
                duration = end_time - start_time
                start_time = time.time()
            logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f Duration: %ds', step, objs.avg, top1.avg, top5.avg,
                         duration)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
