import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch

from model import Net

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')

parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--easy-vit-model', type=str, metavar='E',
                    help="the easy vit model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--hard-vit-model', type=str, metavar='H',
                    help="the hard vit model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--create-easy-hard', type=bool, default=True, metavar='C',
                    help="Whether to split test data")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='O',
                    help="name of the output csv file")

args = parser.parse_args()

if args.create_easy_hard:    
    from preprocess import make_folder
    make_folder(args.data + '/test_images/easy_test_images')
    make_folder(args.data + '/test_images/hard_test_images')

    from detector import detect_test
    detect_test(args)

use_cuda = torch.cuda.is_available()

torch.cuda.empty_cache()

easy_vit_state_dict = torch.load(args.easy_vit_model)
hard_vit_state_dict = torch.load(args.hard_vit_model)

easy_model_vit = Net()
hard_model_vit = Net()

if use_cuda:
    print('Using GPU')
    easy_model_vit.cuda()  
    hard_model_vit.cuda()
else:
    print('Using CPU')

easy_model_vit.load_state_dict(easy_vit_state_dict)

hard_model_vit.load_state_dict(hard_vit_state_dict)
easy_model_vit.eval()
hard_model_vit.eval()

from data import data_transforms

easy_test_dir = args.data + '/test_images/easy_test_images'
hard_test_dir = args.data + '/test_images/hard_test_images'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")

for f in tqdm(os.listdir(easy_test_dir)):
    if 'jpg' in f:
        data = data_transforms['val'](pil_loader(easy_test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()    

        output = easy_model_vit(data)

        pred  = output.data.max(1, keepdim=True)[1]
        output_file.write("%s,%d\n" % (f[:-4], pred))


for f in tqdm(os.listdir(hard_test_dir)):
    if 'jpg' in f:
        data = data_transforms['val'](pil_loader(hard_test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()           
        output = hard_model_vit(data)
       
        pred = output.data.max(1, keepdim=True)[1]
        
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
