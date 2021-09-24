# Assessment-of-glomerular-morphological-patterns-by-deep-learning

Repository to the paper "Assessment of glomerular morphological patterns by deep learning algorithms" published in Journal of Nephrology ().

It contains several scripts to the paper:

#1 topic: In the paper 12 different CNN-classification models (like AlexNet) are trained. To describe the according model architecture, the script "script2plotModels" is added. It plots on basis of hiddenlayer (https://github.com/waleedka/hiddenlayer) the models to a pdf-file (saved in the folder ModelPlots).

#2 topic: Pre-trained models are downloaded and are trained two times on histological images. 
In the first round (1st_TrainingModelOnHistology) the model are re-trained on big set of images defined by one expert alone.
In the second round (2nd_TrainingModelOnHistology) these models are re-loaded and again trained on a set of histological images. This set is significantly smaller and defined on basis of consensus of three expert pathologists.

#3 topic: The models are evaluated on basis of a test set (script2TestTrainedModels) and by using a class-activation maps (https://github.com/chaeyoung-lee/pytorch-CAM) the decision is visualized. 
