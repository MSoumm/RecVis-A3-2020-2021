import os
import shutil
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import Net


def label_data(args, hard_model_name, easy_model_name, bounding_boxes, id_to_class):
    
    # Loading models
    hard_model_dict = torch.load(args.experiment +'/'+ hard_model_name)
    easy_model_dict = torch.load(args.experiment +'/'+ easy_model_name)
    
    use_cuda = torch.cuda.is_available()

    hard_model = Net()
    easy_model = Net()

    if use_cuda:
        hard_model.cuda()
        easy_model.cuda()  

    hard_model.load_state_dict(hard_model_dict)
    hard_model.eval()
    easy_model.load_state_dict(easy_model_dict)
    easy_model.eval()

    from data import data_transforms

    image_dir = args.data + '/self_supervised_images/'
    hard_output_dir = args.data + '/self_supervised/'
    easy_output_dir = args.data + '/easy_self_supervised/'

    def pil_loader(path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    print("Self-supervised learning running...")
    for f in tqdm(os.listdir(image_dir), position=0, leave=True):
        if 'jpg' in f:
            data = data_transforms['val'](pil_loader(image_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()    

            hard_output = hard_model(data)
            hard_output = F.softmax(hard_output)

            hard_prob = hard_output.data.max(1, keepdim=True)[0]
            hard_pred = hard_output.data.max(1, keepdim=True)[1]
            hard_pred = hard_pred.item()

            data = data_transforms['detect'](pil_loader(image_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()    

            box = bounding_boxes[bounding_boxes.index == f[:32]]

            a, b, c, d = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])

            cropped = data[:, :, b:d+b, a:c+a]
            
            cropped = F.interpolate(cropped, size=384)

            easy_output = easy_model(cropped)
            easy_output = F.softmax(easy_output)

            easy_prob = easy_output.data.max(1, keepdim=True)[0]
            easy_pred = easy_output.data.max(1, keepdim=True)[1]
            easy_pred = easy_pred.item()

            # Only label new data if very confident (--> few mis-labels)
            if easy_pred == hard_pred:
                if easy_prob > 0.9:       
                    shutil.copy(image_dir+'/'+f, hard_output_dir+'/'+id_to_class[easy_pred])
                if hard_prob > 0.9:
                    save_image(cropped, easy_output_dir+'/'+id_to_class[easy_pred]+'/'+f)
    
    # Free memory
    del easy_model
    del hard_model
    torch.cuda.empty_cache()

    print("Done!")
