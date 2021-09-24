
#%% background preparation
import torch
from torchvision import models
import hiddenlayer as hl
import torch.nn as nn
from tqdm import tqdm

#%% define the models
model_list = ['alex',
              'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
              'vgg11', 'vgg16', 'vgg19',
              'densenet121', 'squeeznet',
              'inception']

#%% model loading function
def load_model(imodel):
    #Load a pretrained model and reset final fully connected layer.
    if imodel == 'ResNet18':
        model2train = models.resnet18(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet34':
        model2train = models.resnet34(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet50':
        model2train = models.resnet50(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet101':
        model2train = models.resnet101(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'ResNet152':
        model2train = models.resnet152(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg16':
        model2train = models.vgg16(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg19':
        model2train = models.vgg19(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'vgg11':
        model2train = models.vgg11(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'alex':
        model2train = models.alexnet(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'inception':
        model2train = models.inception_v3(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet121':
        model2train = models.densenet121(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet161':
        model2train = models.densenet161(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet169':
        model2train = models.densenet169(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'densenet201':
        model2train = models.densenet201(pretrained=True)
        model2train.modName = imodel + '_loaded '
    if imodel == 'squeeznet':
        model2train = models.squeezenet1_0(pretrained=True)
        model2train.modName = imodel + '_loaded '

    return (model2train)

#%% iterate over the model list, load the models and plot them
for i_model in tqdm(model_list):

    #%% load it
    model2plot = load_model(i_model)

    if i_model == 'inception':
        inputSize = 299
    else:
        inputSize = 224

    model2plot.eval()

    #%% plot it with hiddenlayer
    im = hl.build_graph(model2plot, torch.zeros([1, 3, inputSize, inputSize]))
    im.theme = hl.graph.THEMES["blue"].copy()
    dot = im.build_dot()
    dot.attr("graph", rankdir="TD")
    dot.view("ModelPlots/ModelPlot_" + i_model)
