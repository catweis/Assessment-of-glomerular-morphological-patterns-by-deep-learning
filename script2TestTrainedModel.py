
#%% load the background
from __future__ import print_function, division
import torch
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import torch.nn as nn

#%% define the dataset
data_dir = 'YOUR IMAGE FOLDER PATH'

#%% define the function to get the data
def get_datatransform(inputSize, data_dir):

    data_transforms = {
        dataset2use: transforms.Compose([
            transforms.Resize([inputSize, inputSize]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in [dataset2use]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=False, num_workers=4)
                   for x in [dataset2use]}

    return(data_transforms, image_datasets, dataloaders)

#%% prepare the transformations and the dataset
data_transforms , image_datasets, dataloaders= get_datatransform(259, data_dir)

class_names = dataloaders[dataset2use].dataset.classes
nb_classes = len(class_names)
confusion_matrix = torch.zeros(nb_classes, nb_classes)

#%% visualize the input data
class_names =  ['1', '2', '3',
                '4', '5', '6',
                '7', '8','9']

# legend
# 01 -> normal
# 02 -> amyloidosis
# 03 -> DM
# 04- > global sclerosis
# 05 -> mesangial proliferation
# 06 -> MPGN
# 07 -> necrosis
# 08 -> segmental sclerosis
# 09 -> other structures

df = pd.DataFrame(dataloaders[dataset2use].dataset.samples)
df.columns = ['file', 'class_nr']

df.class_nr = np.array(df.class_nr)

class_labels = ['NaN' for x in range(df.shape[0])]
for i in range(0,df.shape[0]):
    class_labels[i] = class_names[df.class_nr[int(i)]]
df = df.assign(class_labels = class_labels)
sns.set_palette("Set1", n_colors = 12)
sns.countplot(df.class_labels)
plt.xlabel('Pattern')
plt.ylabel('Count [n]')
plt.savefig('DataBase_' + dataset2use + '.jpg')
plt.show()
plt.close()

#%% define the models to load
model_list = ['alex',
              'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
              'vgg11', 'vgg16', 'vgg19',
              'densenet121', 'squeeznet',
              'inception']

accurancy = list(range(0, len(model_list)))
kappa = list(range(0, len(model_list)))
loss = list(range(0, len(model_list)))
print(df)

#%% iterate over the models
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = 'YOUR IMAGE FOLDER PATH'

df_values = pd.DataFrame(list(range(0,len(dataloaders[dataset2use].sampler.data_source.imgs))))

for imodel in model_list:

    #%% prepare the dataset
    if imodel == 'inception':
        inputSize = 299
    else:
        inputSize = 224

    data_transforms, image_datasets, dataloaders = get_datatransform(inputSize, data_dir)

    #%% apply model on test data set (and get a confusion matrix)
    model_ft = torch.load(load_path + '/model_' + imodel + '.pt')
    model_ft.eval()
    vector_prd = []
    vector_exp = []
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders[dataset2use]):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            if i == 0:
                outputs_matrix = outputs
            else:
                outputs_matrix = torch.cat((outputs_matrix, outputs), 0)

            vector_prd = vector_prd + preds.view(-1).cpu().tolist()
            vector_exp = vector_exp + classes.view(-1).cpu().tolist()

            for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    loss_function = nn.CrossEntropyLoss()
    loss_value = loss_function(outputs_matrix.to('cpu'), torch.tensor(vector_exp))
    print(confusion_matrix)

    #%% mount the data into the table
    file_names_list = dataloaders[dataset2use].sampler.data_source.imgs
    file_names = list(range(0, len(file_names_list)))
    for j in range(0, len(file_names_list)):
        t_name = file_names_list[j]
        t_name = str(t_name[0])
        idx = t_name.index('/' +  dataset2use + '/')
        file_names[j] = t_name[int(idx) + 6:len(t_name) - 4]
    df_values['file_names_' + imodel] = file_names
    df_values['exp_' + imodel] = [x+1 for x in vector_exp]
    df_values['pred_' + imodel] = [x+1 for x in vector_prd]

    #%% calcualte the comparison values
    kappa[n] = cohen_kappa_score(vector_prd, vector_exp)
    accurancy[n] = accuracy_score(vector_prd, vector_exp)
    loss[n] = loss_value.tolist()
    print(kappa[n])
    print(accurancy[n])
    n +=1

    #%% plot a confusion matrix
    matrix2plot = confusion_matrix.numpy()
    matrix2plot = matrix2plot.astype(int)
    matrix2plot = normalize(matrix2plot, axis =1, norm = 'l1')
    matrix2plot = matrix2plot.round(decimals=2)
    # create seabvorn heatmap with required labels
    ax = sns.heatmap(matrix2plot,
                     annot = True, linewidths=5, annot_kws={"size": 10},
                     xticklabels=class_names, yticklabels=class_names,
                     cmap = "Blues")
    plt.xlabel('Pattern defined expert #2')
    plt.ylabel('Pattern predicted by ' + imodel)
    plt.savefig(save_path + '/ConfMat_' + imodel + '_' + dataset2use + 'Set.jpg')
    plt.show()
    plt.close()
df_values = df_values.sort_values(by = df_values.columns[1])

#%% save the table to a xlsx-file
df = pd.DataFrame(model_list)
df['accurancy'] = accurancy
df['kappa'] = kappa
df['loss'] = loss

#%%
print(df)
df.to_csv(save_path + '/EvaluationTable_' + dataset2use + 'Set.csv')
df.to_excel(save_path + '/EvaluationTable_' + dataset2use + 'Set.xlsx')
df_values.to_csv(save_path + '/EvaluationTable_values_' + dataset2use + 'Set.csv')
print('exported')

