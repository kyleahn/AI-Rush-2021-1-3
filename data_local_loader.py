from torch.utils import data
from torchvision import transforms
import torch
import os
import random
from PIL import Image
from randaugment import ImageNetPolicy

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except FileNotFoundError as e:
        raise FileNotFoundError(e)


def get_transform(isTrain, random_crop=True, hflip=False):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    transform = []
    before_resolution = 300
    input_resolution = 250
    transform.append(transforms.Resize(before_resolution))
    if isTrain:
        if hflip:
            transform.append(transforms.RandomHorizontalFlip(1.0))
        if random_crop:
            transform.append(transforms.RandomResizedCrop(input_resolution))
            transform.append(ImageNetPolicy())
        else:	
            transform.append(transforms.CenterCrop(input_resolution))
    else:
        transform.append(transforms.CenterCrop(input_resolution))

    transform.append(transforms.ToTensor())
    transform.append(normalize)
    return transforms.Compose(transform)

class TestDataset(data.Dataset):
    def __init__(self, root='../1-3-DATA-fin'):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'test'
        self.data_loc = 'test_data'
        self.path = os.path.join(root, self.data_loc, self.data_idx)

        with open(self.path, 'r') as f:
            lines = f.readlines()

        self.samples = []
        for line in lines:
            idx = line.split("_")[0]
            self.samples.append([line.rstrip('\n'), idx])

        self.transform = get_transform(isTrain=False, random_crop=False)

    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, idx = self.samples[index]
        path = os.path.join(self.root, self.data_loc, path)
        sample = self.transform(pil_loader(path=path))

        return torch.LongTensor([int(idx)]), sample

    def __len__(self):
        return len(self.samples)

class CustomDataset(data.Dataset):
    def __init__(self, is_train=True, root='../1-3-DATA-fin', split=1.0):

        self.root = root
        self.data_idx = 'data_idx'

        self.sample_dir = 'train'
        self.data_loc = 'train_data'
        self.path = os.path.join(root, self.sample_dir, self.data_loc, self.data_idx)

        with open(self.path , 'r') as f:
            lines = f.readlines()

        seed = 1905
        random.seed(seed)
        random.shuffle(lines)

        self.transforms = []
        split = int(len(lines) * split)
        if is_train:
            lines = lines[:split]
            self.transforms.append(get_transform(isTrain=True, random_crop=False, hflip=False))
            self.transforms.append(get_transform(isTrain=True, random_crop=True, hflip=False))
            self.transforms.append(get_transform(isTrain=True, random_crop=True, hflip=False))
            self.transforms.append(get_transform(isTrain=True, random_crop=False, hflip=True))
            self.transforms.append(get_transform(isTrain=True, random_crop=True, hflip=True))
            self.transforms.append(get_transform(isTrain=True, random_crop=True, hflip=True))
        else:
            lines = lines[split:]
            self.transforms.append(get_transform(isTrain=False, random_crop=False, hflip=False, color=False))
        lv_list = [0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 6, 4, 5, 6, 7, 8, 7, 9, 10, 11, 12, 13, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 14, 19, 20, 21, 2, 15, 22, 23, 24, 16, 25, 17, 18, 26, 19, 20, 21, 22, 23, 24, 27, 28, 3, 25, 29, 26, 30, 31, 27, 32, 4, 28, 29, 30, 5, 31, 33, 34, 35, 32, 36, 37, 38, 33, 39, 40, 41, 6, 34, 35, 7, 8, 36, 37, 38, 9, 39, 40, 10, 41, 42, 43, 44, 45, 46, 11, 47, 48, 49, 50, 51, 52]
        self.samples = []
        for line in lines:
            idx = line.split(" ")[0].split("__")[1].split("_")[0]
            label = [v.rstrip('\n') for v in line.split(' ')[1:]]
            label = [lv_list[int(l)] for l in label]
            for i in range(len(self.transforms)):
                self.samples.append([line.split(' ')[0], label, idx, i])


    def __getitem__(self, index):
        '''
        Here, our problem supposes maximum 3 hierarchy
        '''
        path, target, _, transform_idx = self.samples[index]
        path = os.path.join(self.root, self.sample_dir, self.data_loc, path)
        sample = self.transforms[transform_idx](pil_loader(path=path))

        not_exist_label = -100
        if len(target) == 1:
            return torch.LongTensor([index]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([not_exist_label]), torch.LongTensor([not_exist_label])
        elif len(target) == 2:
            return torch.LongTensor([index]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([not_exist_label])
        elif len(target) == 3:
            return torch.LongTensor([index]), sample, torch.LongTensor([int(target[0])]), torch.LongTensor([int(target[1])]), torch.LongTensor([int(target[2])])
        else:
            raise Exception

    def __len__(self):
        return len(self.samples)


def data_loader(root='', dataset=None, phase='train', batch_size=16):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError

    if dataset is None:
        dataset = TestDataset(root=root)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=is_train,
                           drop_last=is_train,
                           num_workers=30 if is_train else 0)
