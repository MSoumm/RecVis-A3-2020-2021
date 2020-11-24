import numpy as np
import PIL

import torchvision.transforms as transforms
from imgaug import augmenters as iaa


class Train_transforms(object):
    # We use imggaug for augmentation 
    def __init__(self):
        self.seq=iaa.Sequential([
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast()
        ])

    def __call__(self, img):
        img=np.array(img)
        img=self.seq.augment_images([img])
        img=PIL.Image.fromarray((img[0]))
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

data_transforms = {
    'detect' : transforms.ToTensor(),

    'train' : transforms.Compose([
        transforms.Resize((384, 384)),
        Train_transforms(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),

    # Without data augmentation for validation
    'val': transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
    }
