%% train_cnn.m
% Baseline CNN training for hyperspectral polymer classification.
% MATLAB R2026b | CPU training
%
% Expected input file:
%   Results/prepared_datastores.mat
%
% This script expects imageDatastore variables named imdsTrain,
% imdsValidation, and imdsTest (as created by prepare_datastores.m).

clear; clc; close all;

%% 1. Locate project folders
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

resultsFolder = fullfile(projectRoot, 'Results');
modelFolder   = fullfile(projectRoot, 'Model');

if ~isfolder(modelFolder)
    mkdir(modelFolder);
end

datastoreFile = fullfile(resultsFolder, 'prepared_datastores.mat');

if ~isfile(datastoreFile)
    error('Could not find %s. Run prepare_datastores.m first.', datastoreFile);
end

%% 2. Load prepared datastores
S = load(datastoreFile);
availableVariables = fieldnames(S);

fprintf('Variables found in prepared_datastores.mat:\n');
disp(availableVariables);

requiredNames = {'imdsTrain', 'imdsValidation', 'imdsTest'};
for k = 1:numel(requiredNames)
    if ~isfield(S, requiredNames{k})
        error(['Expected variable "%s" was not found in prepared_datastores.mat.\n' ...
               'Please share the variable list printed above so we can match the names.'], ...
               requiredNames{k});
    end
end

imdsTrain      = S.imdsTrain;
imdsValidation = S.imdsValidation;
imdsTest       = S.imdsTest; %#ok<NASGU>

% Confirm that labels are available.
if isempty(imdsTrain.Labels) || isempty(imdsValidation.Labels)
    error('Training or validation datastore has no labels.');
end

classNames = categories(removecats(imdsTrain.Labels));
numClasses = numel(classNames);

if numClasses ~= 4
    error('Expected 4 polymer classes, but found %d.', numClasses);
end

fprintf('\nTraining images   : %d\n', numel(imdsTrain.Files));
fprintf('Validation images : %d\n', numel(imdsValidation.Files));
fprintf('Test images       : %d\n', numel(imdsTest.Files));
fprintf('Classes           : %s\n', strjoin(classNames, ', '));

%% 3. Define input dimensions
% These dimensions were established by prepare_datastores.m.
inputSize = [68 73 10];

% Read one image to verify the actual dimensions.
sampleImage = readimage(imdsTrain, 1);
if ~isequal(size(sampleImage), inputSize)
    error('Expected image size [%s], but read [%s].', ...
        num2str(inputSize), num2str(size(sampleImage)));
end
fprintf('\nVerified CNN input size: %d x %d x %d\n', inputSize);

%% 4. Create augmented datastores
% No geometric augmentation is applied in this baseline experiment.
% augmentedImageDatastore packages each image with its categorical label.
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

%% 5. Define the baseline CNN
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

    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
];

fprintf('\nBaseline CNN architecture:\n');
analyzeNetwork(layers);

%% 6. Configure training
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

%% 7. Train the CNN
fprintf('\nStarting baseline CNN training on CPU...\n');
[net, trainingInfo] = trainnet(augTrain, layers, 'crossentropy', options);

%% 8. Save trained model and training information
modelFile = fullfile(modelFolder, 'baseline_cnn.mat');
save(modelFile, 'net', 'trainingInfo', 'layers', 'inputSize', ...
    'classNames', 'miniBatchSize', 'maxEpochs', 'initialLearnRate');

infoFile = fullfile(resultsFolder, 'baseline_training_info.mat');
save(infoFile, 'trainingInfo', 'classNames', 'inputSize');

fprintf('\nTraining completed.\n');
fprintf('Trained model saved to:\n%s\n', modelFile);
fprintf('Training information saved to:\n%s\n', infoFile);

%% 9. Plot and save training curves when available
if isfield(trainingInfo, 'TrainingLoss') && ~isempty(trainingInfo.TrainingLoss)
    figure('Name', 'Baseline CNN Training Loss');
    plot(trainingInfo.TrainingLoss, 'LineWidth', 1.5);
    xlabel('Iteration');
    ylabel('Training loss');
    title('Baseline CNN - Training Loss');
    grid on;
    exportgraphics(gcf, fullfile(resultsFolder, 'baseline_training_loss.png'));
end

if isfield(trainingInfo, 'ValidationLoss') && ~isempty(trainingInfo.ValidationLoss)
    figure('Name', 'Baseline CNN Validation Loss');
    plot(trainingInfo.ValidationLoss, 'LineWidth', 1.5);
    xlabel('Validation iteration');
    ylabel('Validation loss');
    title('Baseline CNN - Validation Loss');
    grid on;
    exportgraphics(gcf, fullfile(resultsFolder, 'baseline_validation_loss.png'));
end

fprintf('\nSTEP 5 COMPLETE: Baseline CNN training script finished.\n');
