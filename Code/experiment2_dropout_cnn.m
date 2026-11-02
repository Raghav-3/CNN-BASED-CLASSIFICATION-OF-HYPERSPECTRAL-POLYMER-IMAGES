%% experiment2_dropout_cnn.m
% Experiment 2: Baseline CNN + dropout regularization.
% MATLAB R2026b | CPU training
%
% Controlled comparison:
%   Same data splits, input size, convolution blocks, optimizer,
%   learning rate, batch size, epochs and validation settings as baseline.
%   The only architecture change is a dropout layer (probability 0.30)
%   after the final ReLU and before global average pooling.
%
% This script uses training and validation data only for training/model
% selection. It does NOT evaluate on the test set.

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

%% 6. Define Experiment 2 CNN
% This is the same baseline architecture with ONE added dropout layer.
dropoutProbability = 0.30;

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

    convolution2dLayer(3, 64, ...
        'Padding', 'same', ...
        'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    dropoutLayer(dropoutProbability, 'Name', 'dropout')

    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
];

fprintf('\nExperiment 2 architecture:\n');
fprintf('Baseline CNN + dropout probability %.2f\n', dropoutProbability);
analyzeNetwork(layers);

%% 7. Use the same training settings as baseline
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

%% 8. Train Experiment 2
fprintf('\nStarting Experiment 2: CNN with dropout on CPU...\n');
[netExperiment2, trainingInfoExperiment2] = ...
    trainnet(augTrain, layers, 'crossentropy', options);

%% 9. Save Experiment 2 model and training information
modelFile = fullfile(modelFolder, 'experiment2_dropout_cnn.mat');
save(modelFile, 'netExperiment2', 'trainingInfoExperiment2', ...
    'layers', 'inputSize', 'classNames', 'dropoutProbability', ...
    'miniBatchSize', 'maxEpochs', 'initialLearnRate');

infoFile = fullfile(resultsFolder, 'experiment2_dropout_training_info.mat');
save(infoFile, 'trainingInfoExperiment2', 'classNames', ...
    'inputSize', 'dropoutProbability');

fprintf('\nExperiment 2 training completed.\n');
fprintf('Model saved to:\n%s\n', modelFile);
fprintf('Training information saved to:\n%s\n', infoFile);

%% 10. Save training loss plots if available
if isfield(trainingInfoExperiment2, 'TrainingLoss') && ...
        ~isempty(trainingInfoExperiment2.TrainingLoss)
    figure('Name', 'Experiment 2 Training Loss');
    plot(trainingInfoExperiment2.TrainingLoss, 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Training loss');
    title('Experiment 2 - CNN with Dropout: Training Loss');
    grid on;
    exportgraphics(gcf, ...
        fullfile(resultsFolder, 'experiment2_training_loss.png'));
end

if isfield(trainingInfoExperiment2, 'ValidationLoss') && ...
        ~isempty(trainingInfoExperiment2.ValidationLoss)
    figure('Name', 'Experiment 2 Validation Loss');
    plot(trainingInfoExperiment2.ValidationLoss, 'LineWidth', 1.5);
    xlabel('Validation iteration');
    ylabel('Validation loss');
    title('Experiment 2 - CNN with Dropout: Validation Loss');
    grid on;
    exportgraphics(gcf, ...
        fullfile(resultsFolder, 'experiment2_validation_loss.png'));
end

fprintf('\nEXPERIMENT 2 COMPLETE.\n');
fprintf('The test set has not been used in this experiment.\n');
