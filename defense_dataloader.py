
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import csv
import os
from PIL import Image
from torchvision import datasets

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize(
        (opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())

    if opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "gtsrb":
        print("GTSRB dataset does not need normalization")
    elif opt.dataset == "imagenet":
        print("ImageNet dataset does not need normalization")
    elif opt.dataset == "pubfig":
        print("PubFig dataset does not need normalization")
    else:
        print(f"Unexpected opt.dataset value: {opt.dataset}")
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class ImageNet(data.Dataset):
    def __init__(self, args, train=True, transform=None):
        super(ImageNet, self).__init__()
        self.args = args
        dataset_dir = os.path.join(args.data_root, args.dataset)
        
        if train:
            self.data_folder = os.path.join(dataset_dir, 'train')
            self.data = datasets.ImageFolder(self.data_folder, transform=transform)
        else:
            self.data_folder = os.path.join(dataset_dir, 'val', 'images')
            self.data = datasets.ImageFolder(self.data_folder, transform=transform)
            
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label
    
class PubFig(data.Dataset):
    def __init__(self, args, train=True, transform=None):
        super(PubFig, self).__init__()
        self.args = args
        dataset_dir = os.path.join(args.data_root, args.dataset)

        self.data_folder = dataset_dir
        self.data = datasets.ImageFolder(self.data_folder, transform=transform)
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        return image, label
    
def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            opt.data_root, train, transform, download=True)
    elif opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform, download=True)
    elif opt.dataset == "imagenet":
        dataset = ImageNet(opt, train, transform)
    elif opt.dataset == "pubfig":
        dataset = PubFig(opt, train, transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            opt.data_root, train, transform, download=True)
    elif opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            opt.data_root, train, transform, download=True)
    elif opt.dataset == "imagenet":
        dataset = ImageNet(opt, train, transform)
    elif opt.dataset == "pubfig":
        dataset = PubFig(opt, train, transform)
    else:
        raise Exception("Invalid dataset")
    return dataset
