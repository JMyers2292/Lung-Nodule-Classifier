
%% Load Data For ML-CNN
clc;
clear all;

PC1filePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';
PC2filePath = 'D:\Jeremy\Desktop\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';

imds = imageDatastore(PC1filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');
%% Splitting The Data 
[imdsTrain, imdsValidation, imdsTesting] = splitEachLabel(imds, 0.70,0.15,'randomize');

%% Multi-Layer NN Layers
MLCNNlayers = 'MLCNN.mat';
load(MLCNNlayers);

%% Training Options- SGDM
options = trainingOptions('sgdm',...
    'Momentum', 0.9,...
    'InitialLearnRate',0.001, ...
    'L2Regularization', 5*10^-4,...
    'MaxEpochs', 250, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

 %% Training ML-CNN
 [net, info] = trainNetwork(imdsTrain, MLCNN, options);
 
%% Loading Pretrained Network & Get Testing Images
clc;
clear all; %#ok<CLALL>

PC1filePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';
PC2filePath = 'D:\Jeremy\Desktop\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';

imds = imageDatastore(PC1filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation, imdsTesting] = splitEachLabel(imds, 0.70, 0.15,'randomize');

%Load Pretrained Multi-Layer CNN network
preTrainedMLCNN = 'trainedMLCNN250.mat';
load(preTrainedMLCNN);

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
