import os
from .dataset_base import DatasetBase


class Middlebury(DatasetBase):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.ref_dir = os.path.join(self.root, 'other-data')
        self.gt_dir = os.path.join(self.root, 'other-gt-interp')
        self.images = []
        # listdir
        for scenes in os.listdir(self.ref_dir):
            tmp_path = []
            tmp_path.append(os.path.join(self.ref_dir, scenes, 'frame10.png'))
            tmp_path.append(os.path.join(self.gt_dir, scenes, 'frame10i11.png'))
            tmp_path.append(os.path.join(self.ref_dir, scenes, 'frame11.png'))
            self.images.append(tmp_path)
