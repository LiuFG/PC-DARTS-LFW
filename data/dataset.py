from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import os.path
import re
import torch
import tarfile
from PIL import Image


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    if class_to_idx is None:
        class_to_idx = dict()
        build_class_idx = True
    else:
        build_class_idx = False
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        if build_class_idx and not subdirs:
            class_to_idx[label] = None
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if build_class_idx:
        classes = sorted(class_to_idx.keys(), key=natural_key)
        for idx, c in enumerate(classes):
            class_to_idx[c] = idx
    images_and_targets = zip(filenames, [class_to_idx[l] for l in labels])  # image names and labels
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    if build_class_idx:
        return images_and_targets, classes, class_to_idx
    else:
        return images_and_targets # image names and labels


class Dataset(data.Dataset):
    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None):

        imgs, _, _ = find_images_and_targets(root)  # image names and labels
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]




def default_loader(path): #change: add
    #img = Image.open(path).convert('L') #L for gray scale
    img = Image.open(path)
    return img

def default_list_reader(fileList):  #change: add
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():

            str= line.strip().split(' ')
            if len(str)==2:
                imgPath=str[0]
                label = str[1]
            else: # imPath includes spaces
                imgPath=line.strip()[:-2].strip()
                label=line.strip()[-1]
            # imgPath=line.strip()[:-2].strip()
            # label=line.strip()[-1]

            imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):  #change: add
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

    # def filenames():
    #     return self.imgList







def _extract_tar_info(tarfile):  # I think not relevant
    class_to_idx = {}
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        class_to_idx[label] = None
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    for idx, c in enumerate(sorted(class_to_idx.keys(), key=natural_key)):
        class_to_idx[c] = idx
    tarinfo_and_targets = zip(files, [class_to_idx[l] for l in labels])
    tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets


class DatasetTar(data.Dataset): # I think not relevant

    def __init__(self, root, load_bytes=False, transform=None):

        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.imgs = _extract_tar_info(tf)
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.imgs[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

