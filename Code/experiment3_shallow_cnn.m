%% experiment3_shallow_cnn.m
% Experiment 3: Shallower CNN with TWO convolution blocks.
% MATLAB R2026b | CPU training
%
% Controlled comparison against baseline:
%   Same data splits, preprocessing, input size, optimizer, learning rate,
%   mini-batch size, maximum epochs, and validation settings.
%   The only intended architecture change is removing the third
%   convolution + batch-normalization + ReLU block.
%
% This script uses training and validation data only. It does NOT use
% the test datastore or evaluate test performance.

clear; clc; close all;

%% 1. Locate project folders
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

resultsFolder = fullfile(projectRoot, 'Results');
modelFolder = fullfile(projectRoot, 'Model');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

datastoreFile = fullfile(resultsFolder, 'prepared_datastores.mat');
if ~isfile(datastoreFile)
    error('Prepared datastores not found: %s', datastoreFile);
end

%% 2. Load training and validation datastores
S = load(datastoreFile);

requiredNames = {'imdsTrain', 'imdsValidation'};
for k = 1:numel(requiredNames)
    if ~isfield(S, requiredNames{k})
        error('Variable "%s" not found in prepared_datastores.mat.', ...
            requiredNames{k});
    end
end

imdsTrain = S.imdsTrain;
imdsValidation = S.imdsValidation;

classNames = categories(removecats(imdsTrain.Labels));
numClasses = numel(classNames);

if numClasses ~= 4
    error('Expected 4 classes; found %d.', numClasses);
end

fprintf('Training images   : %d\n', numel(imdsTrain.Files));
fprintf('Validation images : %d\n', numel(imdsValidation.Files));
fprintf('Classes           : %s\n', strjoin(classNames, ', '));

%% 3. Verify input dimensions
inputSize = [68 73 10];
sampleImage = readimage(imdsTrain, 1);

if ~isequal(size(sampleImage), inputSize)
    error('Expected input [%s], but read [%s].', ...
        num2str(inputSize), num2str(size(sampleImage)));
end

fprintf('Verified input size: %d x %d x %d\n', inputSize);

%% 4. Prepare datastores
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% 5. Set random seed for reproducibility
rng(42, 'twister');

%% 6. Define the shallower CNN (two convolution blocks)
% Compared with baseline, the third convolution block is removed.
layers = [
    imageInputLayer(inputSize, ...
        'Name', 'input', ...
        'Normalization', 'none')

    convolution2dLayer(3, 16, ...
        'Padding', 'same', ...
        'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, ...
        'Stride', 2, ...
        'Name', 'pool1')

    convolution2dLayer(3, 32, ...
        'Padding', 'same', ...
        'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, ...
        'Stride', 2, ...
        'Name', 'pool2')

    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
];

fprintf('\nExperiment 3 architecture:\n');
fprintf('Shallower CNN with two convolution blocks.\n');
analyzeNetwork(layers);

%% 7. Keep training settings consistent with baseline
miniBatchSize = 16;
maxEpochs = 40;
initialLearnRate = 1e-3;
validationFrequency = max(1, floor(numel(imdsTrain.Files) / miniBatchSize));

options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augValidation, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', 8, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 15, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 8. Train Experiment 3
fprintf('\nStarting Experiment 3: shallower two-block CNN on CPU...\n');
[netExperiment3, trainingInfoExperiment3] = ...
    trainnet(augTrain, layers, 'crossentropy', options);

%% 9. Save Experiment 3 model and training information
modelFile = fullfile(modelFolder, 'experiment3_shallow_cnn.mat');
save(modelFile, 'netExperiment3', 'trainingInfoExperiment3', ...
    'layers', 'inputSize', 'classNames', ...
    'miniBatchSize', 'maxEpochs', 'initialLearnRate');

infoFile = fullfile(resultsFolder, 'experiment3_shallow_training_info.mat');
save(infoFile, 'trainingInfoExperiment3', 'classNames', 'inputSize');

fprintf('\nExperiment 3 training completed.\n');
fprintf('Model saved to:\n%s\n', modelFile);
fprintf('Training information saved to:\n%s\n', infoFile);

%% 10. Save training loss plots if available
if isfield(trainingInfoExperiment3, 'TrainingLoss') && ...
        ~isempty(trainingInfoExperiment3.TrainingLoss)
    figure('Name', 'Experiment 3 Training Loss');
    plot(trainingInfoExperiment3.TrainingLoss, 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Training loss');
    title('Experiment 3 - Shallow CNN: Training Loss');
    grid on;
    exportgraphics(gcf, ...
        fullfile(resultsFolder, 'experiment3_training_loss.png'));
end

if isfield(trainingInfoExperiment3, 'ValidationLoss') && ...
        ~isempty(trainingInfoExperiment3.ValidationLoss)
    figure('Name', 'Experiment 3 Validation Loss');
    plot(trainingInfoExperiment3.ValidationLoss, 'LineWidth', 1.5);
    xlabel('Validation iteration');
    ylabel('Validation loss');
    title('Experiment 3 - Shallow CNN: Validation Loss');
    grid on;
    exportgraphics(gcf, ...
        fullfile(resultsFolder, 'experiment3_validation_loss.png'));
end

fprintf('\nEXPERIMENT 3 COMPLETE.\n');
fprintf('The test set has not been used in this experiment.\n');
