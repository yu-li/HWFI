import numpy as np
from PIL import Image
import torch.utils.data as data


class DatasetBase(data.Dataset):
    def __getitem__(self, index):
        sample = self.loader(self.images[index])
        img_shape = sample['image'][0].shape[:2]
        sample['ishape'] = img_shape

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.images)

    def loader(self, scene_path):
        sample = {'image': [], 'input_files': []}
        # read frame
        for _frame_file in scene_path:
            _frame = np.array(Image.open(_frame_file).convert('RGB')).astype(np.float32)
            sample['image'].append(_frame)
            sample['input_files'].append(_frame_file)
        return sample
