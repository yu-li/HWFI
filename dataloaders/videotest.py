import os
from .dataset_base import DatasetBase


class VideoTest(DatasetBase):
    def __init__(self, root, transform=None):
        self.mov_dir = root
        self.transform = transform
        self.sample_list = []
        # listdir
        for frame in os.listdir(self.mov_dir):
            self.sample_list.append(os.path.join(self.mov_dir, frame))
        self.sample_list.sort()

    def __getitem__(self, index):
        sample = self.loader([self.sample_list[index], self.sample_list[index + 1]])
        img_shape = sample['image'][0].shape[:2]
        sample['ishape'] = img_shape
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_list) - 1
