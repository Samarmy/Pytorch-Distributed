import os.path
import random
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class BranchedTemporalDataset(data.Dataset):

    def __init__(self):

        self.dir_AB = os.path.abspath("/s/chopin/k/grad/sarmst/CR/stgan/train_data/combined/train")
        self.AB_paths = sorted(self.make_dataset(self.dir_AB))  # get image paths
        self.input_nc = 3
        self.output_nc = 3

    def __getitem__(self, index):

        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B

        w, h = AB.size
        w4 = int(w / 4)
        A_0 = AB.crop((0, 0, w4, h))
        A_1 = AB.crop((w4, 0, 2*w4, h))
        A_2 = AB.crop((2*w4, 0, 3*w4, h))
        B = AB.crop((3*w4, 0, w, h))

        # apply the same transform to both A and B
        transform_params = self.get_params(A_0.size)
        A_transform = self.get_transform(transform_params, grayscale=(self.input_nc == 1))
        B_transform = self.get_transform(transform_params, grayscale=(self.output_nc == 1))

        A_0 = A_transform(A_0)
        A_1 = A_transform(A_1)
        A_2 = A_transform(A_2)
        B = B_transform(B)

        return {'A_0': A_0, 'A_1': A_1, 'A_2': A_2, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def get_params(self, size):
        w, h = size
        new_h = h
        new_w = w
        new_h = new_w = 286

        x = random.randint(0, np.maximum(0, new_w - 256))
        y = random.randint(0, np.maximum(0, new_h - 256))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}


    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        # for root, _, fnames in sorted(os.walk(dir))[rangeStart:rangeEnd]:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def get_transform(self, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in "resize_and_crop":
            osize = [286, 286]
            transform_list.append(transforms.Resize(osize, method))
        elif 'scale_width' in "resize_and_crop":
            transform_list.append(transforms.Lambda(lambda img: self.__scale_width(img, 286, method)))

        if 'crop' in "resize_and_crop":
            if params is None:
                transform_list.append(transforms.RandomCrop(256))
            else:
                transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], 256)))

        if "resize_and_crop" == 'none':
            transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base=4, method=method)))

        if not False:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))

        if convert:
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __make_power_2(self, img, base, method=Image.BICUBIC):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if (h == oh) and (w == ow):
            return img

        self.__print_size_warning(ow, oh, w, h)
        return img.resize((w, h), method)

    def __scale_width(self, img, target_width, method=Image.BICUBIC):
        ow, oh = img.size
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
        return img.resize((w, h), method)

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __print_size_warning(self, ow, oh, w, h):
        """Print warning information about image size(only print once)"""
        if not hasattr(self.__print_size_warning, 'has_printed'):
            print("The image size needs to be a multiple of 4. "
                  "The loaded image size was (%d, %d), so it was adjusted to "
                  "(%d, %d). This adjustment will be done to all images "
                  "whose sizes are not multiples of 4" % (ow, oh, w, h))
            self.__print_size_warning.has_printed = True

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
