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
 
%% Loading Pretrained Network & Get Testing Images
clc;
clear all; %#ok<CLALL>

PC1filePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\Inception Images';
PC2filePath = 'D:\Jeremy\Desktop\Dropbox\Coding\MATLAB_code\Assignment 3\Inception Images';
imds = imageDatastore(PC1filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

%Split Testing Images into 150
[imdsTrain, imdsValidation, imdsTesting] = splitEachLabel(imds, 0.70, 0.15,'randomize');

%Load Pretrained inceptionv3 network
pretrainedNet = 'trainedInception250.mat';
load(pretrainedNet);

%% Testing Pretrained CNN Showing Confusion Matrix
%Will use pretrained network to classify images from imdsTesting
labels = classify(net, imdsTesting);

confMat = confusionmat(imdsTesting.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

%% Showing Test Results Of CNN
labels = classify(net, imdsTesting);

% Randomly go through 150 imdsValidation images and comparing them to the
% predicted labels given to them.
numbers = [];
for i = 1:10
    rand = randi(150);
    im = imread(imdsTesting.Files{rand});
    im = imresize(im,20);
    axis = subplot(2,5,i);
    imshow(im)
    if labels(rand) == imdsTesting.Labels(rand)
        colorText = 'g';
        title(char(labels(rand)),'Color',colorText);
    else
        colorText = 'r';
        title(char(labels(rand)),'Color',colorText);
    end
    numbers = [numbers rand]; %#ok<AGROW>
end
