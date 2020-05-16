from __future__ import print_function
from tqdm import tqdm
import os
import torch
from torch import nn
torch.backends.cudnn.bencmark = True
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import PIL
from PIL import Image
import argparse
import numpy as np
import torchvision
from torchvision import transforms
from .multi_process import multi_process
# outNpyFile = open("./result/test1.npy","w+")

#k-fold cross validation（k-折叠交叉验证）
#将n份数据分为n_folds份，以次将第i份作为测试集，其余部分作为训练集
def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train, test])
    return folds

#求解当前阈值时的准确率
def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

#eval_acc和find_best_threshold共同工作，来求试图找到最佳阈值
def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def prepare_tensor(img, input_size=144, crop_size=128):
    val_transform = transforms.Compose([
        transforms.Resize(input_size),#144*144
        transforms.CenterCrop(crop_size),#144*144>>128*128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return val_transform(img)



def validate(args,  net):
    #model_to_load = args.model
    # net =model
    # num_classes = args.NO_classes
    # in_features = net.fc.in_features
    # net.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    # 加载网络
    # net = torch.nn.DataParallel(net).cuda()
    net.eval()

    # # discard last fc??????????????
   # net.module.fc = Identity()

    predicts = list()

    with open(args.val_list) as f:           #Python引入了with语句来自动帮我们调用close()方法：
        pairs_lines = f.readlines()[1:]

    image_folder = args.val_folder

    multi_process(pairs_lines, net, predicts, image_folder, lfw_predict)

    accuracy = []#准确率
    thd = [] #最佳阈值
    folds = KFold(n=6000, n_folds=10, shuffle=False)     #k-fold cross validation（k-折叠交叉验证）folds的形式为[[train,test],[train,test].....]
    thresholds = np.arange(-1.0, 1.0, 0.005)    #取数组为-1到1，步长为0.005
    predicts = np.array(list(map(lambda line:line.strip('\n').split(), predicts)))

    for idx, (train, test) in enumerate(folds):
        # predicts[train/test]形式为：
        # [['Doris_Roberts/Doris_Roberts_0001.jpg'
        # 'Doris_Roberts/Doris_Roberts_0003.jpg' '0.6532696413605743' '1'],.....]
        #print ('in folder '+str(idx)+'\n')
        best_thresh = find_best_threshold(thresholds, predicts[train]) #寻找最佳阈值
        accuracy.append(eval_acc(best_thresh, predicts[test]))#通过上面的得到的最佳阈值来对test数据集进行测试得到准确率
        thd.append(best_thresh) #thd阈值
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    # np.mean：计算均值，np.std：计算标准差
    # 输出结果分别为：准确率均值，准确率标准差，阈值均值
    return np.mean(accuracy), np.std(accuracy), np.mean(thd)

def lfw_predict(pairs_lines, net, predicts, image_folder, count):
    p = pairs_lines[count].replace('\n', '').split('\t')
    if 3 == len(p):
        sameflag = 1
        ##形式例如：Woody_Allen/Woody_Allen_0002.jpg
        gg = p[0]
        name1 = p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2 = p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
    if 4 == len(p):
        sameflag = 0
        name1 = p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
        name2 = p[2] + '_' + '{:04}.jpg'.format(int(p[3]))

    img1 = Image.open(os.path.join(image_folder, name1))
    img2 = Image.open(os.path.join(image_folder, name2))

    img1_flip = img1.transpose(PIL.Image.FLIP_LEFT_RIGHT)  # 左右翻转
    img2_flip = img2.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    img_tensors = [prepare_tensor(x, 144, 128) for x in [img1, img1_flip, img2, img2_flip]]
    img_tensors = [torch.unsqueeze(x, 0).cuda() for x in img_tensors]  # unsqueeze扩维
    img_tensors = torch.cat(img_tensors, dim=0)  # torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。

    features = net(img_tensors, lfw=True)  # two return values
    features = features.data.cpu().numpy()

    f1 = np.concatenate([features[0, :], features[1, :]])
    f2 = np.concatenate([features[2, :], features[3, :]])

    f1 = f1.reshape(1, -1)
    f2 = f2.reshape(1, -1)

    cosdistance = cosine_similarity(f1, f2)  # 计算X和Y中样本间的余弦相似度
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, cosdistance[0][0], sameflag))