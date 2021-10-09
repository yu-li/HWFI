import os
from .dataset_base import DatasetBase


class UCF101(DatasetBase):
    def __init__(self, root, transform=None):
        self.ref_dir = root
        self.transform = transform
        self.images = []
        # listdir
        for scenes in os.listdir(self.ref_dir):
            tmp_path = []
            tmp_path.append(os.path.join(self.ref_dir, scenes, 'frame_00.png'))
            tmp_path.append(os.path.join(self.ref_dir, scenes, 'frame_01_gt.png'))
            tmp_path.append(os.path.join(self.ref_dir, scenes, 'frame_02.png'))
            self.images.append(tmp_path)
