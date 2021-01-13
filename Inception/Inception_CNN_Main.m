%% Load Data For Inception CNN
clc;
clear all; %#ok<CLALL>

PC1filePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\Inception Images';
PC2filePath = 'D:\Jeremy\Desktop\Dropbox\Coding\MATLAB_code\Assignment 3\Inception Images';

%% Splitting The Data 
imds = imageDatastore(PC2filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85,'randomize');
%% Inceptionv3 NN
net = inceptionv3;
layers = net.Layers;
lgraph = layerGraph(net);

%% Changing Last Layers Of Inception
newLearnableLayer = fullyConnectedLayer(2, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',2, ...
    'BiasLearnRateFactor',2);
lgraph = replaceLayer(lgraph,'predictions',newLearnableLayer);

newSMLayer = softmaxLayer('Name','new_sm');
lgraph = replaceLayer(lgraph,'predictions_softmax',newSMLayer);

newClassifierLayer = classificationLayer('Name','new_cl');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassifierLayer);

%% Training Options- SGDM
options = trainingOptions('sgdm',...
    'Momentum', 0.9,...
    'InitialLearnRate',0.001, ...
    'L2Regularization', 5*10^-4,...
    'MaxEpochs', 50, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Training Inceptionv3 CNN
 [net, info] = trainNetwork(imdsTrain, lgraph, options);
 
 %% Load Pretrained inceptionv3 network
pretrainedNet = 'trainedInception250.mat';
load(pretrainedNet);

 %% Validating CNN
correctClassifications = 0;
incorrectClassifications = 0;

labels = classify(net, imdsValidation);

% Randomly go through 150 imdsValidation images and comparing them to the
% predicted labels given to them.
for i = 1:150
    rand = randi(150);
    im = imread(imdsValidation.Files{rand});
    if labels(rand) == imdsValidation.Labels(rand)
        correctClassifications = correctClassifications + 1;
    else
        incorrectClassifications = incorrectClassifications + 1;
    end
end

accuracy = correctClassifications /sum(correctClassifications + incorrectClassifications)
confMat = confusionmat(imdsValidation.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))