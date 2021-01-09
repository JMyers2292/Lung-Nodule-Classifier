
%% Load And Split Data
clc;
clear all;

PC1filePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';
imds = imageDatastore(PC1filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');


PC2filePath = 'D:\Jeremy\Desktop\Dropbox\Coding\MATLAB_code\Assignment 3\LungImages';
imds = imageDatastore(PC2filePath,'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.85,'randomize');

%% Multi-Layer NN Layers
MLCNN = 'MLCNN.mat';
load(MLCNN);

%% Inceptionv3 NN
inception = inceptionv3;

%% Iniciate Inceptionv3 Layers
inceptionLayers = inception.Layers;

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
%  [net, info] = trainNetwork(imdsTrain, MLCNN, options);
%  
 %% Training Inceptionv3 CNN
 [net, info] = trainNetwork(imdsTrain, inceptionLayers, options);
%% Validating CNN

correctClassifications = 0;
incorrectClassifications = 0;

labels = classify(net, imdsValidation);

% Randomly go through 150 imdsValidation images and comparing them to the
% predicted labels given to them.
for i = 1:150
    rand = randi(150);
    im = imread(imdsValidation.Files{rand});
%     im = imresize(im,20);
%     imshow(im);
    if labels(rand) == imdsValidation.Labels(rand)
%         colorText = 'g';
        correctClassifications = correctClassifications + 1;
    else
%         colorText = 'r';
        incorrectClassifications = incorrectClassifications + 1;
    end
end
% title(char(labels(ii)),'Color',colorText);
accuracy = correctClassifications /sum(correctClassifications + incorrectClassifications);
confMat = confusionmat(imdsValidation.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

