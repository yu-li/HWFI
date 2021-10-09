import cv2
import numpy as np
import numpy.random as random
import torch


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for idx in range(0, len(sample['image'])):
                tmp_image = sample['image'][idx]
                tmp_image = cv2.flip(tmp_image, flipCode=1)
                sample['image'][idx] = tmp_image
        return sample

    def __str__(self):
        return "RandomHorizontalFlip"


class RandomVerticalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            for idx in range(0, len(sample['image'])):
                tmp_image = sample['image'][idx]
                tmp_image = cv2.flip(tmp_image, flipCode=0)
                sample['image'][idx] = tmp_image
        return sample

    def __str__(self):
        return "RandomVerticalFlip"


class RandomReverseTemporalOrder(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'].reverse()
        return sample

    def __str__(self):
        return "RandomReverseTemporalOrder"


class RandomCrop(object):
    def __init__(self, resolution=None):
        self.resolution = resolution

    def __call__(self, sample):
        if self.resolution is None:
            return sample

        tmp_image = sample['image'][0]
        h, w = tmp_image.shape[:2]
        cx_min = 0
        cx_max = w - self.resolution[1]
        cy_min = 0
        cy_max = h - self.resolution[0]
        assert (cx_max >= 0) and (cy_max >= 0)

        x = random.random_integers(cx_min, cx_max)
        y = random.random_integers(cy_min, cy_max)

        for idx in range(0, len(sample['image'])):
            tmp_image = sample['image'][idx]
            tmp_image = tmp_image[y:y + self.resolution[0], x:x + self.resolution[1]]
            sample['image'][idx] = tmp_image
        return sample

    def __str__(self):
        return "RandomCrop:" + str(self.resolution)


class ToTensor(object):
    def __call__(self, sample):

        for elem in sample.keys():
            if 'image' in elem:
                for idx in range(0, len(sample[elem])):
                    tmp = sample[elem][idx].astype(np.float32)
                    if tmp.ndim == 2:
                        tmp = tmp[:, :, np.newaxis]
                    # swap color axis
                    tmp = tmp.transpose((2, 0, 1))
                    sample[elem][idx] = torch.from_numpy(tmp)
        return sample

    def __str__(self):
        return "ToTensor"


class Normalize(object):
    def __call__(self, sample):
        for idx in range(len(sample['image'])):
            sample["image"][idx] = (sample["image"][idx] / 255.0)

        return sample


class PadImage(object):
    def __init__(self, stride):
        self.stride = stride

    def pad_images(self, images):
        height, width, _ = images[0].shape
        image_count = len(images)
        if (height % self.stride) != 0:
            new_height = (height // self.stride + 1) * self.stride
            for i in range(image_count):
                images[i] = np.pad(images[i], (
                    (0, new_height - height),
                    (0, 0),
                    (0, 0),
                ), 'reflect')

        if (width % self.stride) != 0:
            new_width = (width // self.stride + 1) * self.stride
            for i in range(image_count):
                images[i] = np.pad(images[i], ((0, 0), (0, new_width - width), (0, 0)), 'reflect')
        return images

    def __call__(self, sample):
        self.pad_images(sample['image'])
        return sample
