
#%% load the background
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import seaborn as sns

#%% define the dataset and get the weigths
data_dir = 'YOUR IMAGE FOLDER PATH'

#%% define function for preparing the database
def prep_database(inputSize):

    #%%
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([inputSize,inputSize]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([inputSize,inputSize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    #%% get the dataset
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    #%% define the sampler
    targets = torch.Tensor(image_datasets['train'].targets).long()
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=samples_weight,
        num_samples=len(samples_weight),
        replacement=True)

    #%% define the dataloader
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
                                                        batch_size=16,
                                                        sampler=sampler,
                                                        num_workers=4),
                   'val': torch.utils.data.DataLoader(image_datasets['val'],
                                                      batch_size=16,
                                                      shuffle=True,
                                                      num_workers=4)}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return(dataloaders, dataset_sizes, class_names)

#%% fill the dataloader
dataloaders, dataset_sizes, class_names = prep_database(224)

# checks if GPU is available, and then decide accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print('GPU-mode is set')
else:
    print('CPU-mode is set')


#%% show the image data
# Get a batch of training data
inputs, classes = next(iter(dataloaders['val']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

pwd = os.getcwd()+'/Pytorch_GlomerulusClassification'
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, pwd)
from visFunctions import imshow

imshow(out, title=[class_names[x] for x in classes])

#%% define the training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

#%% define the model loading function
model_folder_path= 'FOLDER TO YOUR TRAINED MODELS' # in subfolder trainedModels should be the models to load
load_path = model_folder_path + "/trainedModels"
save_path = model_folder_path + "/re_trainedModels"

def load_model(imodel):
    model2train = torch.load(load_path + '/model_' + imodel + '.pt')
    return (model2train)

#%% define a list of models
model_list = ['alex',
              'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
              'vgg11', 'vgg16', 'vgg19',
              'densenet121', 'squeeznet',
              'inception']
n_classes = len(class_names)

#%% iterate over the model list
for imodel in model_list:

    #%% # Data augmentation and normalization for training
    # Just normalization for validation
    if imodel == 'inception':
        inputSize = 299
    else:
        inputSize = 224

    (dataloaders, dataset_sizes, class_names) = prep_database(inputSize)

    #%% test dataloader
    test_input, test_output = next(iter(dataloaders['val']))

    #%% load the model
    model2train = load_model(imodel)
    print(model2train._get_name() + " is loaded for re-training")

    #%% adapt the model
    model_ft = model2train  # for debugging re-loading can be avoided
    model_ft = model_ft.to('cuda')

    #%% set the training parameter
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #%% train the model
    model_ft = model_ft.to(device)
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=50)

    #%% test the model
    inputs = test_input.to(device)
    classes = test_output.to(device)
    outputs = model_ft(inputs)
    _, test_preds = torch.max(outputs, 1)

    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(classes.view(-1), test_preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    matrix2plot = confusion_matrix.numpy()
    matrix2plot = matrix2plot.astype(int)

    ax = sns.heatmap(matrix2plot,
                     annot=True, linewidths=5, annot_kws={"size": 10},
                     xticklabels=class_names, yticklabels=class_names,
                     cmap="Blues")
    plt.xlabel('Pattern expected')
    plt.ylabel('Pattern predicted')
    plt.title('Test for ' + imodel)
    plt.show()

    #%% save the model
    torch.save(model_ft, save_path + '/model_' + imodel + '.pt')


