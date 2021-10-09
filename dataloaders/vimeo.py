import os
from .dataset_base import DatasetBase


class VIMEO(DatasetBase):
    def __init__(self, root, split='eval', transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        self.images = []
        if self.split == 'train':
            self.list = os.path.join(self.root, 'tri_trainlist.txt')
        else:
            self.list = os.path.join(self.root, 'tri_testlist.txt')
        # listdir
        with open(self.list, 'r') as f:
            for line in f.readlines():
                subdir = line.strip('\n')
                if '/' in subdir:
                    tmp_path = []
                    tmp_path.append(os.path.join(self.root, 'sequences', subdir, 'im1.png'))
                    tmp_path.append(os.path.join(self.root, 'sequences', subdir, 'im2.png'))
                    tmp_path.append(os.path.join(self.root, 'sequences', subdir, 'im3.png'))
                    self.images.append(tmp_path)
