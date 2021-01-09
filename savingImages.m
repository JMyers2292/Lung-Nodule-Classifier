%% Labeling Images and Saving Into Correspoding Folders
clc;
clear all;

%Designated Folder Paths For Images
benignFolderPath = 'C:\Users\Jeremy\Desktop\A3 CNN Data\LungImage\LungImages\benign';
malignantFolderPath = 'C:\Users\Jeremy\Desktop\A3 CNN Data\LungImage\LungImages\malignant';


%File Path For .mat Files to be Used for Image Converstion and Labeling
imageFilePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\LungImage\gene_features.mat';
labelFilePath = 'C:\Users\Jeremy\Dropbox\Coding\MATLAB_code\Assignment 3\LungImage\label.mat';

%Designated Folder For Inceptionv3 Resized Images
incepBenignFP = 'C:\Users\Jeremy\Desktop\A3 CNN Data\LungImage\Inception Images\benign';
incepMaligFP = 'C:\Users\Jeremy\Desktop\A3 CNN Data\LungImage\Inception Images\malignant';

%Loading The .mat Files
load(imageFilePath);
load(labelFilePath);

%Specifying the Classes
benign = [0 1];
malignant = [1 0];
intermediate = [0 0];

%% Using A For Loop To Go Through Each Image and Match The Label To It
%Initiating Count For Classes
numBenign = 0;
numMalignant = 0;
for i = 1:1000
   %Converts .mat File Into Grey Scale Lung Image
   image = gene_features(i,:,:);
   image = reshape(image, [28, 28]);
   image = mat2gray(image);
   
   %Checks The Correspoding Lung Image With Its Label
   if label(i,:) == benign
    numBenign = numBenign + 1; 
    baseFileName = sprintf('benign.%d.png',numBenign);
    filePath = fullfile(benignFolderPath,baseFileName);
    imwrite(image, filePath);
   end
   if label(i,:) == malignant
    numMalignant = numMalignant + 1;
    baseFileName = sprintf('malignant.%d.png',numMalignant);
    filePath = fullfile(malignantFolderPath,baseFileName);
    imwrite(image, filePath);
   end
end
disp('Done Converting Images And Classifying Them');

%% Converting images to 299x299 for Inceptionv3 and saving
%Initiating Count For Classes
numBenign = 0;
numMalignant = 0;
for i = 1:1000
   %Converts .mat File Into Grey Scale Lung Image
   image = gene_features(i,:,:);
   image = reshape(image, [28, 28]);
   image = mat2gray(image);
   rs_image = imresize(image, [299 299]);
   
   %Checks The Correspoding Lung Image With Its Label
   if label(i,:) == benign
    numBenign = numBenign + 1; 
    baseFileName = sprintf('benign.%d.png',numBenign);
    filePath = fullfile(incepBenignFP,baseFileName);
    imwrite(rs_image, filePath);
   end
   if label(i,:) == malignant
    numMalignant = numMalignant + 1;
    baseFileName = sprintf('malignant.%d.png',numMalignant);
    filePath = fullfile(incepMaligFP,baseFileName);
    imwrite(rs_image, filePath);
   end
end
disp('Done Converting Images And Classifying Them');
