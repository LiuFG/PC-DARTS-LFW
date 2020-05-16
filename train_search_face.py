import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
# import torch.nn.functional as F
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
from torch.autograd import Variable
from model_search_face import Network
from architect import Architect
from data import create_loader, ImageList, FastCollateMixup, resolve_data_config
from mfcode.lfw_eval import validate as lfw_eval
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser("face")
parser.add_argument('--data', type=str, default='/home/liufg/my_work/data_augment/face/webface_crop/train', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.0, help='drop path probability')
parser.add_argument('--dropout_type', type=str, default='none', help=' end | cell | None')
parser.add_argument('--save', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--data_portion', type=float, default=0.1, help='portion of data')
parser.add_argument('--arch_epoch', type=int, default=15, help='portion of data')
parser.add_argument('--valid_epoch', type=int, default=20, help='portion of data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--root_path', type=str, default='/home/liufg/my_work/data_augment/face/webface_crop/train',
                    metavar='H', help='Dir Head')
parser.add_argument('--trainFile', type=str, default='trainlist_full.txt', metavar='TRF', help='training file name')
# parser.add_argument('--testFile', type=str, default='test.txt', metavar='TEF',
#                     help='test file name')
parser.add_argument('--val_list', default='/home/liufg/my_work/data_augment/face/LFW/pairs.txt', type=str)
parser.add_argument('--val_folder', default='/home/liufg/my_work/data_augment/face/LFW/aligned', type=str)
parser.add_argument('--val_imlist', default='/home/liufg/my_work/data_augment/face/LFW/image_list.txt', type=str)
parser.add_argument('--prefetcher', action='store_true', default=True,
                    help='disable fast prefetcher')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
# Augmentation parameters
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--color-jitter', type=float, default=0.0, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')  # change: to 0, original: 0.4
parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                    help='Random erase prob (default: 0.)')  # 0 means switch off ?? to check
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')   # 0.2 to switch on
parser.add_argument('--mixup-off-epoch', default=60, type=int, metavar='N',
                    help='turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='label smoothing (default: 0.1)')   ### change: label smoothing, default: 0.1

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
args = parser.parse_args()

args.save = './res/search/{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10576
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.dropout_type)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # train_transform, valid_transform = utils._data_transforms_cifar10(args)
  # if args.set=='cifar100':
  #     train_data = dset.CIFAR100(root=args.data, train=True, download=False, transform=train_transform)
  # else:
  #     train_data = dset.CIFAR10(root=args.data, train=True, download=False, transform=train_transform)

  dataset_train = ImageList(root=args.data, fileList=args.data + '/' + args.trainFile)

  collate_fn = None
  if args.prefetcher and args.mixup > 0:
    collate_fn = FastCollateMixup(args.mixup, args.smoothing, args.num_classes)
  num_train = len(dataset_train)
  # indices = list(range(num_train))
  # randint(args.data_portion * num_train)

  indices = np.linspace(0, num_train-1, args.data_portion*num_train, dtype=np.int)
  random.shuffle(indices)
  num_train = len(indices)
  # patch = int(np.floor(args.data_portion * num_train))
  split = int(np.floor(args.train_portion * num_train))

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
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    collate_fn=collate_fn,
    auto_augment=args.aa
  )
  valid_queue = create_loader(
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
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    collate_fn=collate_fn,
    auto_augment=args.aa
  )

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  writer = SummaryWriter(os.path.join(args.save, 'tensorboard'))
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect,
                                 criterion, optimizer, lr, epoch, args.arch_epoch, writer)
    logging.info('train_acc %f', train_acc)
    writer.add_scalar('lr', lr, epoch)
    writer.add_scalar('train_acc', train_acc, epoch)
    writer.add_scalar('train_loss', train_obj, epoch)
    # validation
    # if (epoch+1) >= args.valid_epoch and (epoch+1) % 5 == 0:
    #   LFWACC, std, thd = lfw_eval(args, model)
    #   logging.info('lfw_eval LFW_ACC:%f LFW_std:%f LFW_thd:%f', LFWACC,std,thd)
    #   writer.add_scalar('LFW_ACC', LFWACC, epoch)
    #   writer.add_scalar('LFW_std', std, epoch)
    #   writer.add_scalar('LFW_thd', thd, epoch)

    # utils.save(model, os.path.join(args.save, 'weights.pt'))
  writer.close()


def train(train_queue, valid_queue, model, architect, criterion,
          optimizer, lr, epoch, arch_epoch, writer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    total_step = epoch*len(train_queue)+step
    model.train()
    n = input.size(0)
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    # input = Variable(input, requires_grad=False).cuda()
    # target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda(non_blocking=True)
    target_search = target_search.cuda(non_blocking=True)
    # input_search = Variable(input_search, requires_grad=False).cuda()
    # target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    if epoch >= arch_epoch:
      if step % args.report_freq == 0:
        big_path_list = utils.get_big_path_list(4)
        path_dict_list = utils.arch_params2dict([model.alphas_normal,model.betas_normal], 4)
        for idx, dict in enumerate(path_dict_list):
          big_path_name = big_path_list[idx]
          writer.add_scalars(big_path_name + '_ori', dict['ori'], total_step)
          writer.add_scalars(big_path_name + '_res', dict['res'], total_step)
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

