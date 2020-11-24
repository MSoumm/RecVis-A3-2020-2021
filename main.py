import argparse
import os
import shutil
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.datasets import ImageFolder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RecVis A3 training script')
    parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                        help="folder where data is located. train_images/ and val_images/ need to be found in the folder")

    parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                        help='input batch size for training (default: 16)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--self-supervised', type=bool, default=True, metavar='SS',
                        help='whether to use self-supervised data augmentation (default: True)')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='L',
                        help='how many batches to wait before logging training status (default: 10)')

    parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                        help='folder where experiment outputs are located.')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)

    self_supervised = args.self_supervised

    #import preprocess
    from preprocess import make_folders, get_bounding_boxes, shuffle, make_self_supervised

    # Create usefull folders and reshuffle dataset
    make_folders(args)
    shuffle(args)

    if self_supervised:
        # Download self-supervised dataset
        make_self_supervised(args)
    
    # Create bounding boxes for original dataset
    from detector import detect
    detect(args) 

    # Data initialization and loading
    from data import data_transforms

    print("Loading images...")
    easy_train_loader_dataset = ImageFolder(args.data + '/easy_train_images',
                             transform=data_transforms['train'])

    # class <-> id mappings                                     
    class_to_id = easy_train_loader_dataset.class_to_idx
    id_to_class = {v: k for k, v in class_to_id.items()}  

    easy_train_loader = torch.utils.data.DataLoader(easy_train_loader_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    easy_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/easy_val_images',
                             transform=data_transforms['val']),
        batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    hard_train_dataset = datasets.ImageFolder(args.data + '/train_images',
                             transform=data_transforms['train'])

    hard_train_loader = torch.utils.data.DataLoader(hard_train_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    hard_val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                             transform=data_transforms['val']),
        batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    print("Done !")

    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    import model
    from model import Net

    def train(epoch, model, loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                    100. * batch_idx / len(loader), loss.data.item()))

    def validation(model, loader):
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(loader.dataset),
            100. * correct / len(loader.dataset)))
        
        return  100. * correct / len(loader.dataset)

    def run_model(model, name, train_loader, val_loader, optimizer):
        correct = 0
        best_epoch = 0
        # We keep track of the best model on val set

        for epoch in range(1, args.epochs + 1):
            train(epoch, model, train_loader, optimizer)
            val_score = validation(model, val_loader)
            if val_score > correct:
                correct = val_score
                best_epoch = epoch

            model_file = args.experiment + '/'+name+'_model_' + str(epoch) + '.pth'   
            if name == 'self_supervised':
                torch.save(model, model_file)
            else:
                torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
        return best_epoch

    print("Running 1st stage model...")
    
    vit_model = Net()
    if use_cuda:
        print('Using GPU')
        vit_model.cuda()
    else:
        print('Using CPU')

    # Pretrained elements have a smaller lr then fresh ones
    optimiser = optim.Adam([{'params': vit_model.model.blocks.parameters()},
                           {'params': vit_model.model.head.parameters(), 'lr': 1e-4},
                           {'params': vit_model.fc.parameters(), 'lr': 1e-4}],
                           lr=1e-5)
    
    best_hard_epoch = run_model(vit_model, 'hard_vit', hard_train_loader, hard_val_loader, optimiser)
    best_easy_epoch = run_model(vit_model, 'easy_vit', easy_train_loader, easy_val_loader, optimiser)
    
    print("Done")
    print("Best 1st stage hard model : " + str(best_hard_epoch))
    print("Best 1st stage easy model : " + str(best_easy_epoch))

    # Free memory
    del vit_model
    torch.cuda.empty_cache()

    if self_supervised:
        import evaluate_unlabeled
        from evaluate_unlabeled import label_data

        best_hard_model = 'hard_vit_model_'+str(best_hard_epoch)+'.pth'
        best_easy_model = 'easy_vit_model_'+str(best_easy_epoch)+'.pth'
        
        # Boudning boxes of self suppervised dataset
        bounding_boxes = get_bounding_boxes(args)
    
        # Create pseudo-labels for self-supervised dataset
        label_data(args, best_hard_model, best_easy_model, bounding_boxes, id_to_class)

        # Copy files from original dataset to the augmented one
        print("Moving images...")
        for label in os.listdir(args.data + '/train_images'):
            for image in os.listdir(args.data + '/train_images/'+label):
                shutil.copy(args.data + '/train_images/'+label+'/'+image, args.data +'/self_supervised/'+label)

        for label in os.listdir(args.data + '/easy_train_images'):
            for image in os.listdir(args.data + '/easy_train_images/'+label):
                shutil.copy(args.data + '/easy_train_images/'+label+'/'+image, args.data +'/easy_self_supervised/'+label)
        print("Done !")

        def get_class_distribution(dataset_obj):
            count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
            
            for element in dataset_obj:
                y_lbl = element[1]
                y_lbl = id_to_class[y_lbl]
                count_dict[y_lbl] += 1
                    
            return count_dict

        print("Loading images...")
        self_supervised_dataset = datasets.ImageFolder(args.data + '/self_supervised/',
                                transform=data_transforms['train'])

        # the new dataset is unbalanced so we need a weighted sampler
        def make_sampler(dataset):
            target_list = torch.tensor(dataset.targets)
            class_count = [i for i in get_class_distribution(dataset).values()]
            class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
            class_weights_all = class_weights[target_list]

            weighted_sampler = torch.utils.data.WeightedRandomSampler(
                weights=class_weights_all,
                num_samples=len(class_weights_all),
                replacement=True
            )
            return weighted_sampler

        hard_weighted_sampler = make_sampler(self_supervised_dataset)
        final_hard_train_loader = torch.utils.data.DataLoader(dataset=self_supervised_dataset,
            batch_size=args.batch_size, sampler=hard_weighted_sampler)


        easy_self_supervised_dataset = datasets.ImageFolder(args.data + '/easy_self_supervised/',
                                transform=data_transforms['train'])
        easy_weighted_sampler = make_sampler(easy_self_supervised_dataset)

        final_easy_train_loader =  torch.utils.data.DataLoader(dataset=easy_self_supervised_dataset,
            batch_size=args.batch_size, sampler=easy_weighted_sampler)

        print("Done !")

        vit_model = Net()
        if use_cuda:
            print('Using GPU')
            vit_model.cuda()
        else:
            print('Using CPU')

        optimiser = optim.Adam([{'params': vit_model.model.blocks.parameters()},
                            {'params': vit_model.model.head.parameters(), 'lr': 1e-4},
                            {'params': vit_model.fc.parameters(), 'lr': 1e-4}],
                            lr=1e-5)
        
        print("Running 2nd stage model... ")
        best_hard_model = run_model(vit_model, 'final_hard_vit', final_hard_train_loader, hard_val_loader, optimiser)
        best_easy_model = run_model(vit_model, 'final_easy_vit', final_easy_train_loader, easy_val_loader, optimiser)
        print("Done")

        print("Best hard model : " + str(best_hard_model))
        print("Best easy model : " + str(best_easy_model))
