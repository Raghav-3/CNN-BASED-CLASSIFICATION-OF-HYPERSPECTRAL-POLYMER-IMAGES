%% predict_new_polymer.m
% Predict the polymer class of one new hyperspectral ENVI image.
% MATLAB R2026b
%
% Select a matching .bil and .hdr file when prompted.
% Uses the saved baseline CNN and the exact training preprocessing settings.
%
% IMPORTANT:
% - The input must contain 168 bands and be no larger than 68 x 73.
% - Images larger than the trained input size are rejected; they are NOT cropped.
% - For a genuinely independent demo, use a sample not used in model training,
%   validation, or the final test evaluation.

clear; clc; close all;

%% 1. Locate project folders
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);
resultsFolder = fullfile(projectRoot, 'Results');
modelFolder = fullfile(projectRoot, 'Model');

modelFile = fullfile(modelFolder, 'baseline_cnn.mat');
datastoreFile = fullfile(resultsFolder, 'prepared_datastores.mat');

if ~isfile(modelFile)
    error('Baseline model not found: %s', modelFile);
end
if ~isfile(datastoreFile)
    error('Prepared preprocessing configuration not found: %s', datastoreFile);
end

%% 2. Load trained model and preprocessing configuration
M = load(modelFile, 'net', 'classNames', 'inputSize');
D = load(datastoreFile, 'preprocessingConfig');

net = M.net;
classNames = M.classNames;
inputSize = M.inputSize;
cfg = D.preprocessingConfig;

requiredFields = {'selectedBands','bandMean','bandStd', ...
    'targetHeight','targetWidth','inputSize','paddingValue'};
for k = 1:numel(requiredFields)
    if ~isfield(cfg, requiredFields{k})
        error('Missing preprocessingConfig field: %s', requiredFields{k});
    end
end

if ~isequal(inputSize, cfg.inputSize)
    error('Model input size and saved preprocessing input size do not match.');
end

selectedBands = cfg.selectedBands;
bandMean = cfg.bandMean;
bandStd = cfg.bandStd;
targetHeight = cfg.targetHeight;
targetWidth = cfg.targetWidth;
paddingValue = cfg.paddingValue;

if numel(selectedBands) ~= inputSize(3) || ...
        numel(bandMean) ~= inputSize(3) || numel(bandStd) ~= inputSize(3)
    error('Band selection and normalization statistics do not match CNN depth.');
end
if any(~isfinite(bandStd)) || any(bandStd <= 0)
    error('Invalid saved training standard deviations.');
end

fprintf('Loaded baseline CNN.\n');
fprintf('Expected input: %d x %d x %d\n', inputSize);
fprintf('Selected original bands: %s\n', mat2str(selectedBands));

%% 3. Select the new ENVI image files
[bilName, bilFolder] = uigetfile({'*.bil','ENVI BIL files (*.bil)'}, ...
    'Select the new hyperspectral BIL file');

if isequal(bilName, 0)
    fprintf('No BIL file selected. Prediction cancelled.\n');
    return;
end

bilPath = fullfile(bilFolder, bilName);
[~, baseName] = fileparts(bilPath);
hdrPath = fullfile(bilFolder, [baseName '.hdr']);

if ~isfile(hdrPath)
    error('Matching HDR file not found: %s', hdrPath);
end

%% 4. Read metadata and hyperspectral cube using MATLAB functions
fprintf('\nReading hyperspectral image...\n');
hdrInfo = enviinfo(hdrPath);
hcube = imhypercube(bilPath);
cube = gather(hcube);

[H, W, numBands] = size(cube);
fprintf('Original cube size: %d x %d x %d\n', H, W, numBands);

if numBands ~= 168
    error('Expected 168 spectral bands, but this image has %d.', numBands);
end

% Confirm that selected indices are valid for this cube.
if any(selectedBands < 1) || any(selectedBands > numBands)
    error('Saved selected band indices are incompatible with this image.');
end

% Metadata consistency check when the properties are available.
if isfield(hdrInfo, 'Bands') && hdrInfo.Bands ~= numBands
    error('HDR metadata reports %d bands, but the loaded cube has %d.', ...
        hdrInfo.Bands, numBands);
end

%% 5. Reject images larger than target size (never crop)
if H > targetHeight || W > targetWidth
    error(['This image is %d x %d, larger than the trained target size ' ...
        '%d x %d. It cannot be processed under the required padding-only ' ...
        'policy. No cropping or resizing was performed.'], ...
        H, W, targetHeight, targetWidth);
end

%% 6. Select the same 10 PCA-selected original bands
selectedCube = single(cube(:, :, selectedBands));

%% 7. Normalize using training-only statistics
normalizedCube = zeros(H, W, numel(selectedBands), 'single');

for b = 1:numel(selectedBands)
    band = selectedCube(:, :, b);
    normalizedCube(:, :, b) = ...
        (band - single(bandMean(b))) ./ single(bandStd(b));
end

if any(~isfinite(normalizedCube(:)))
    error('Normalized image contains NaN or Inf values.');
end

%% 8. Apply the same balanced symmetric padding used during training
% Padding is applied after normalization. The saved paddingValue is used.
padHeight = targetHeight - H;
padWidth = targetWidth - W;

padTop = floor(padHeight / 2);
padBottom = padHeight - padTop;
padLeft = floor(padWidth / 2);
padRight = padWidth - padLeft;

processedImage = ones(targetHeight, targetWidth, inputSize(3), 'single') ...
    .* single(paddingValue);

rowRange = (padTop + 1):(padTop + H);
colRange = (padLeft + 1):(padLeft + W);
processedImage(rowRange, colRange, :) = normalizedCube;

fprintf('Padding [top bottom left right]: [%d %d %d %d]\n', ...
    padTop, padBottom, padLeft, padRight);
fprintf('Processed input size: %d x %d x %d\n', size(processedImage));

%% 9. Predict using the baseline CNN
fprintf('\nClassifying image...\n');
scores = minibatchpredict(net, processedImage, ...
    'ExecutionEnvironment', 'cpu');

predictedLabel = scores2label(scores, categorical(classNames));
predictedLabel = string(predictedLabel);

scoreValues = gather(extractdata(scores));
scoreValues = double(scoreValues(:));

% Convert scores to percentages for display. For a softmax output these
% are class probabilities; they are not a guarantee of calibrated certainty.
scorePercent = 100 .* scoreValues;

%% 10. Display prediction and class scores
fprintf('\n========== POLYMER PREDICTION ==========\n');
fprintf('Image: %s\n', bilName);
fprintf('Predicted polymer: %s\n', predictedLabel);
fprintf('\nClass scores:\n');

scoreTable = table(string(classNames(:)), scorePercent, ...
    'VariableNames', {'PolymerClass','ScorePercent'});
scoreTable = sortrows(scoreTable, 'ScorePercent', 'descend');
disp(scoreTable);

%% 11. Visualize prediction
figure('Name', 'New Polymer Prediction');

subplot(1, 2, 1);
% Display three selected bands as a false-colour composite.
rgbBands = min(3, size(selectedCube, 3));
if rgbBands == 3
    rgbImage = selectedCube(:, :, [3 2 1]);
else
    rgbImage = repmat(selectedCube(:, :, 1), 1, 1, 3);
end
rgbImage = mat2gray(rgbImage);
imshow(rgbImage);
title('Selected-band false-colour view');

subplot(1, 2, 2);
bar(categorical(scoreTable.PolymerClass), scoreTable.ScorePercent);
ylabel('Score (%)');
xlabel('Polymer class');
title(['Predicted class: ' char(predictedLabel)]);
ylim([0 100]);
grid on;

%% 12. Save prediction record
predictionRecord = table(string(bilName), predictedLabel, ...
    max(scorePercent), ...
    'VariableNames', {'ImageFile','PredictedPolymer','TopScorePercent'});

predictionCsv = fullfile(resultsFolder, 'unseen_image_prediction.csv');
writetable(predictionRecord, predictionCsv);

fprintf('\nPrediction record saved to:\n%s\n', predictionCsv);
fprintf('PREDICTION DEMO COMPLETED.\n');
