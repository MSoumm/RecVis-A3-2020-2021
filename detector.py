import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image

import torch
from torchvision.utils import save_image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import datasets

from data import data_transforms


#
# Detects birds in the dataset then crops the images
#

def detect(args):
    print("Loading detector...")
    maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
    if torch.cuda.is_available():
        maskrcnn.cuda()
    maskrcnn.eval()
    print("Loaded !\n")

    print("Detecting birds ...\n")
    
    def sort_images(loader, folder, mapping):
        name = 0
        for data, target in tqdm(loader, leave=True, position=0):
            results = maskrcnn(data.cuda())
            for e, result in enumerate(results):
                boxes = result['boxes'].tolist()    # Bounding boxes
                labels =  result['labels'].tolist() # Labels
                scores = result['scores'].tolist()  # Confidence associated with bounding box

                # Keep only bird labels and boxes (label 16 in COCO)
                only_bird_boxes = np.array([boxes[i] for i in range(len(boxes)) if labels[i] == 16])
                only_birds_scores= np.array([scores[i] for i in range(len(boxes))  if labels[i] == 16])
                
                # if low confidence -> hard image
                if only_bird_boxes.size == 0 or only_birds_scores.max() < 0.85:   
                    pass
                else : 
                    try:
                        i = np.argmax(only_birds_scores)
                        box = only_bird_boxes[i]

                        a, b, c, d = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                        # Crop image on bird
                        cropped = data[e, :, b:d, a:c]

                        save_image(cropped, folder+"/"+mapping[target[e].item()]+"/"+str(name)+".png", format ="png")
                    except ValueError:
                        # Bounding box outside image (very rare)
                        pass
            name += 1

    train_dataset = ImageFolder(args.data + '/train_images',
                                        transform=data_transforms['detect'])
    # class <-> id mapping                                     
    class_to_id = train_dataset.class_to_idx
    id_to_class = {v: k for k, v in class_to_id.items()}  
    
    preprocess_train_loader = DataLoader(train_dataset,batch_size=1, 
                                         num_workers=1, shuffle=True)
            
    preprocess_val_loader = DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms['detect']), batch_size=1,  
                             num_workers=1)

    sort_images(preprocess_train_loader, args.data + "/easy_train_images", id_to_class)
    sort_images(preprocess_val_loader, args.data + "/easy_val_images", id_to_class)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def detect_test(args):
    print("Loading detector...")
    maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
    if torch.cuda.is_available():
        maskrcnn.cuda()
    maskrcnn.eval()
    print("Loaded !\n")

    for f in tqdm(os.listdir(args.data +'/test_images/mistery_category')):
        if 'jpg' in f:
            data = data_transforms['detect'](pil_loader(args.data +'/test_images/mistery_category/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda() 
            
            results = maskrcnn(data.cuda())

            for e, result in enumerate(results):
                boxes = result['boxes'].tolist()    # Bounding boxes
                labels =  result['labels'].tolist() # Labels
                scores = result['scores'].tolist()  # Confidence associated with bounding box

                # Keep only bird labels and boxes (label 16 in COCO)
                only_bird_boxes = np.array([boxes[i] for i in range(len(boxes)) if labels[i] == 16])
                only_birds_scores= np.array([scores[i] for i in range(len(boxes))  if labels[i] == 16])
                
                # if low confidence -> hard image
                if only_bird_boxes.size == 0 or only_birds_scores.max() < 0.85:   
                    shutil.copy(args.data +'/test_images/mistery_category/'+f, args.data+'/test_images/hard_test_images')
                else : 
                    try:
                        i = np.argmax(only_birds_scores)
                        box = only_bird_boxes[i]

                        a, b, c, d = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                        # Crop image on bird
                        cropped = data[e, :, b:d, a:c]

                        shutil.copy(args.data +'/test_images/mistery_category/'+f, args.data+'/test_images/easy_test_images')
                    except ValueError:
                        # Bounding box outside image (very rare)
                        pass
