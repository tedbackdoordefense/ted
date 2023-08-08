
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x

def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())

    if opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)
 

def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


def get_dataset(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    else:
        raise Exception("Invalid dataset")
    return dataset
