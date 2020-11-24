import os
import shutil
import glob
import random
from tqdm import tqdm
import tarfile
import wget
import pandas as pd

def make_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def make_folders(args):

    #Craeate usefull folders
    make_folder(args.experiment)

    make_folder(args.data + '/easy_train_images')
    make_folder(args.data + '/easy_val_images')

    for name in os.listdir(args.data +'/train_images'):
        make_folder(args.data + '/easy_train_images/'+name)
        make_folder(args.data + '/easy_val_images/'+name)

    if args.self_supervised:
        make_folder(args.data + '/self_supervised')
        make_folder(args.data + '/easy_self_supervised')

        for name in os.listdir(args.data +'/train_images'):
            make_folder(args.data + '/self_supervised/'+name)
            make_folder(args.data + '/easy_self_supervised/'+name)

def shuffle(args):

    # Re-shuffle the dataset #
    for name in os.listdir(args.data + '/val_images'):
        images = glob.glob(args.data + "/val_images/"+name+"/*.jpg")
        for image in images:
            shutil.move(image, args.data + "/train_images/"+name+"/")

        random_images = random.sample(glob.glob(args.data + "/train_images/"+name+"/*.jpg"), len(images))
        for image in random_images:
            shutil.move(image,args.data + "/val_images/"+name+"/")

def make_self_supervised(args):
    print("Downloading self-supervised dataset (this may take a while)...")
    
    url = "https://www.dropbox.com/s/nf78cbxq6bxpcfc/nabirds.tar.gz?dl=1"  # dl=1 is important
    wget.download(url) 
    
    print("Done !")

    f = tarfile.open('nabirds.tar.gz', 'r:gz')

    # Mapping of our 20 classes to NABird classes
    mapping = {416 : "009.Brewer_Blackbird",
               268 : "010.Red_winged_Blackbird",
               404 : "011.Rusty_Blackbird",
               306 : "012.Yellow_headed_Blackbird",
               254 : "013.Bobolink",
               706 : "014.Indigo_Blunting",
               687 : "015.Lazuli_Blunting",
               723 : "016.Painted_Blunting",
               275 : "019.Gray_Catbird",
               745 : "020.Yellow_breasted_Chat",
               889 : "021.Eastern_Towhee",
               75  : "023.Brandt_Cormorant",
               590 : "026.Bronzed_Cowbird",
               435 : "028.Brown_Creeper",
               683 : "029.Americain_Crow",
               710 : "030.Fish_Crow",
               139 : "031.Black_billed_Cuckoo",
               77  : "033.Yellow_billed_Cuckoo",
               255 : "034.Gray_crowned_Rosy_Finch"
               }

    print("Extracting...")
    files = f.getnames()
    f.extract('nabirds/bounding_boxes.txt', args.data)
    f.extract('nabirds/image_class_labels.txt', args.data)
    f.extract('nabirds/hierarchy.txt', args.data)

    keys = mapping.keys()

    ###
    # Helper dataframes to extract files
    img_to_cat = pd.read_csv(args.data + '/nabirds/image_class_labels.txt', sep= ' ', header=0, names = ['id', 'class'])
    img_to_cat['id'] = img_to_cat['id'].apply(lambda name: name.replace("-", ""))
    img_to_cat = img_to_cat.set_index('id')

    cat_to_id = pd.read_csv(args.data+'/nabirds/hierarchy.txt', sep= ' ', header=0, names = ['class', 'superclass'])
    cat_to_id = cat_to_id.set_index('class')
    ###

    # Extracting images
    for image in tqdm(files, position = 0, leave = True):
        if 'nabirds/images/' in image:
            cat = int(image[15:19])
            if cat in keys:
                f.extract(image, args.data)
            else:
                super_class = int(cat_to_id[cat_to_id.index == cat]['superclass'])
                if super_class in keys:
                    f.extract(image, args.data)
    print("Done !")

    os.remove('nabirds.tar.gz')

    print("Mixing images...")
    make_folder(args.data + '/self_supervised_images')
    for cat in tqdm(os.listdir(args.data + '/nabirds/images')):
        for img in os.listdir(args.data + '/nabirds/images/'+cat):
            shutil.move(args.data + '/nabirds/images/'+cat+'/'+img, args.data+'/self_supervised_images/')
    print("Done !")

def get_bounding_boxes(args):
    bounding_boxes = pd.read_csv(args.data + "/nabirds/bounding_boxes.txt", sep=" ", header=0, names = ['file', 'x1', 'y1', 'x2', 'y2'])
    bounding_boxes['file'] = bounding_boxes['file'].apply(lambda name: name.replace("-", ""))
    bounding_boxes = bounding_boxes.set_index('file')
    
    return bounding_boxes
