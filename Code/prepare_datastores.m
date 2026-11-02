%% prepare_datastores.m
% Step 4: Apply the common PCA bands, normalize using TRAINING data only,
% pad images (padding only), and create train/validation/test imageDatastores.
%
% Prerequisites:
%   Results/dataset_inventory.mat
%   Results/dataset_splits.mat
%   Results/pca_band_selection.mat
%
% Run:
%   run('Code/prepare_datastores.m')

clc;

%% 1. Locate project and load saved results
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);
resultsRoot = fullfile(projectRoot, 'Results');

inventoryFile = fullfile(resultsRoot, 'dataset_inventory.mat');
splitFile = fullfile(resultsRoot, 'dataset_splits.mat');
pcaFile = fullfile(resultsRoot, 'pca_band_selection.mat');

if ~isfile(inventoryFile)
    error('Missing dataset_inventory.mat. Run inspect_dataset.m first.');
end
if ~isfile(splitFile)
    error('Missing dataset_splits.mat. Run create_data_splits.m first.');
end
if ~isfile(pcaFile)
    error('Missing pca_band_selection.mat. Run select_common_bands_pca.m first.');
end

A = load(inventoryFile, 'inventory', 'classNames');
B = load(splitFile, 'trainTable', 'validationTable', 'testTable');
C = load(pcaFile, 'selectedBands');

inventory = A.inventory;
classNames = A.classNames;
trainTable = B.trainTable;
validationTable = B.validationTable;
testTable = B.testTable;
selectedBands = C.selectedBands(:)';

numSelectedBands = numel(selectedBands);
if numSelectedBands ~= 10
    error('Expected exactly 10 selected bands, found %d.', numSelectedBands);
end

%% 2. Establish target dimensions using image dimensions only
% Use all image dimensions, not class labels, to find a size that can
% contain every original image without cropping or resizing.
targetHeight = max(inventory.Height);
targetWidth = max(inventory.Width);

fprintf('Target padded image size: %d x %d x %d\n', ...
    targetHeight, targetWidth, numSelectedBands);

if any(inventory.Height > targetHeight) || any(inventory.Width > targetWidth)
    error('Target dimensions are smaller than at least one original image.');
end

%% 3. Estimate per-band normalization statistics from TRAINING pixels only
% Statistics are calculated before padding, so artificial padding pixels
% do not affect the training mean or standard deviation.
fprintf('\nCalculating normalization statistics from training images only...\n');

sumValues = zeros(1, numSelectedBands, 'double');
sumSquares = zeros(1, numSelectedBands, 'double');
numValidPixels = 0;

for i = 1:height(trainTable)

    bilPath = char(trainTable.BILPath(i));
    hdrPath = char(trainTable.HDRPath(i));

    hdr = enviinfo(hdrPath);
    hcube = imhypercube(bilPath);
    cube = gather(hcube);

    [H, W, B] = size(cube);
    if H ~= hdr.Height || W ~= hdr.Width || B ~= hdr.Bands
        error('HDR/cube dimension mismatch for %s.', bilPath);
    end
    if any(selectedBands < 1) || any(selectedBands > B)
        error('Selected band index is outside the available range for %s.', bilPath);
    end

    selectedCube = double(cube(:, :, selectedBands));
    spectra = reshape(selectedCube, [], numSelectedBands);

    validRows = all(isfinite(spectra), 2);
    spectra = spectra(validRows, :);

    if isempty(spectra)
        error('No finite pixel spectra found in %s.', bilPath);
    end

    sumValues = sumValues + sum(spectra, 1);
    sumSquares = sumSquares + sum(spectra.^2, 1);
    numValidPixels = numValidPixels + size(spectra, 1);

    if mod(i, 20) == 0 || i == height(trainTable)
        fprintf('Normalization scan: %d / %d training images.\n', ...
            i, height(trainTable));
    end
end

bandMean = sumValues ./ numValidPixels;

% Sample standard deviation (N-1 denominator).
if numValidPixels > 1
    varianceNumerator = sumSquares - (sumValues.^2 ./ numValidPixels);
    bandVariance = max(varianceNumerator ./ (numValidPixels - 1), 0);
else
    error('Not enough valid training pixels to calculate standard deviation.');
end

bandStd = sqrt(bandVariance);

% Protect against constant or nearly constant bands.
smallStd = bandStd < 1e-12;
if any(smallStd)
    warning('Some selected bands have near-zero standard deviation. Setting their std to 1.');
    bandStd(smallStd) = 1;
end

fprintf('\nTraining-only normalization statistics:\n');
disp(table(selectedBands(:), bandMean(:), bandStd(:), ...
    'VariableNames', {'BandIndex', 'TrainingMean', 'TrainingStd'}));

%% 4. Create imageDatastores with one shared preprocessing reader
% The ReadFcn applies the same:
%   - 10 selected original band indices
%   - training-derived per-band mean and standard deviation
%   - target image size
%   - balanced padding
% to all three subsets.
readFcn = @(filename) readAndPreprocessHyperspectral( ...
    filename, selectedBands, bandMean, bandStd, ...
    targetHeight, targetWidth);

trainFiles = cellstr(trainTable.BILPath);
validationFiles = cellstr(validationTable.BILPath);
testFiles = cellstr(testTable.BILPath);

imdsTrain = imageDatastore(trainFiles, ...
    'FileExtensions', '.bil', ...
    'ReadFcn', readFcn);

imdsValidation = imageDatastore(validationFiles, ...
    'FileExtensions', '.bil', ...
    'ReadFcn', readFcn);

imdsTest = imageDatastore(testFiles, ...
    'FileExtensions', '.bil', ...
    'ReadFcn', readFcn);

% Assign labels in the same order as the file lists.
imdsTrain.Labels = categorical(trainTable.Class, classNames);
imdsValidation.Labels = categorical(validationTable.Class, classNames);
imdsTest.Labels = categorical(testTable.Class, classNames);

%% 5. Verify datastore sizes and class counts
fprintf('\nDatastore observation counts:\n');
fprintf('Training   : %d\n', numel(imdsTrain.Files));
fprintf('Validation : %d\n', numel(imdsValidation.Files));
fprintf('Test       : %d\n', numel(imdsTest.Files));

fprintf('\nTraining class counts:\n');
disp(countEachLabel(imdsTrain));

fprintf('Validation class counts:\n');
disp(countEachLabel(imdsValidation));

fprintf('Test class counts:\n');
disp(countEachLabel(imdsTest));

%% 6. Inspect one preprocessed observation
sampleImage = preview(imdsTrain);
sampleSize = size(sampleImage);

fprintf('\nPreprocessed sample size: %s\n', mat2str(sampleSize));

if numel(sampleSize) ~= 3 || ...
        sampleSize(1) ~= targetHeight || ...
        sampleSize(2) ~= targetWidth || ...
        sampleSize(3) ~= numSelectedBands
    error('Preprocessed sample has an unexpected size.');
end

if ~isa(sampleImage, 'single')
    warning('Preprocessed image is not single precision; actual class is %s.', ...
        class(sampleImage));
end

%% 7. Save preprocessing configuration and datastores
% Datastores contain ReadFcn handles. Save them for convenience, but
% recreate them by rerunning this script if MATLAB cannot restore a handle.
preprocessingConfig = struct();
preprocessingConfig.selectedBands = selectedBands;
preprocessingConfig.bandMean = bandMean;
preprocessingConfig.bandStd = bandStd;
preprocessingConfig.targetHeight = targetHeight;
preprocessingConfig.targetWidth = targetWidth;
preprocessingConfig.inputSize = [targetHeight, targetWidth, numSelectedBands];
preprocessingConfig.paddingMethod = 'Balanced symmetric zero padding after normalization';
preprocessingConfig.paddingValue = 0;
preprocessingConfig.normalization = 'Per-band z-score using training pixels only';
preprocessingConfig.numValidTrainingPixels = numValidPixels;
preprocessingConfig.classNames = classNames;

save(fullfile(resultsRoot, 'preprocessing_config.mat'), ...
    'preprocessingConfig', 'selectedBands', 'bandMean', 'bandStd', ...
    'targetHeight', 'targetWidth', 'classNames');

save(fullfile(resultsRoot, 'prepared_datastores.mat'), ...
    'imdsTrain', 'imdsValidation', 'imdsTest', ...
    'preprocessingConfig', '-v7.3');

fprintf('\nPreprocessing and datastore preparation completed successfully.\n');
fprintf('CNN input size: %d x %d x %d\n', ...
    targetHeight, targetWidth, numSelectedBands);
fprintf('Saved preprocessing_config.mat and prepared_datastores.mat in Results.\n');

%% Local function: read, select bands, normalize and pad one image
function outputImage = readAndPreprocessHyperspectral( ...
    bilPath, selectedBands, bandMean, bandStd, targetHeight, targetWidth)

    [folder, baseName] = fileparts(bilPath);
    hdrPath = fullfile(folder, [baseName, '.hdr']);

    hdr = enviinfo(hdrPath);
    hcube = imhypercube(bilPath);
    cube = gather(hcube);

    [H, W, B] = size(cube);

    if H ~= hdr.Height || W ~= hdr.Width || B ~= hdr.Bands
        error('HDR/cube dimension mismatch for %s.', bilPath);
    end

    if H > targetHeight || W > targetWidth
        error(['Image exceeds the established target size. ' ...
               'Inference/preprocessing stopped for: %s'], bilPath);
    end

    if any(selectedBands < 1) || any(selectedBands > B)
        error('A selected band index is invalid for: %s', bilPath);
    end

    % Select the same ten original bands for every image.
    selectedCube = double(cube(:, :, selectedBands));

    % Normalize using statistics calculated from training pixels only.
    meanArray = reshape(bandMean, 1, 1, []);
    stdArray = reshape(bandStd, 1, 1, []);
    normalizedCube = (selectedCube - meanArray) ./ stdArray;

    if any(~isfinite(normalizedCube(:)))
        error('Non-finite values encountered after normalization in %s.', bilPath);
    end

    % Balanced padding: any odd extra pixel is placed on bottom/right.
    extraRows = targetHeight - H;
    extraCols = targetWidth - W;

    padTop = floor(extraRows / 2);
    padBottom = extraRows - padTop;
    padLeft = floor(extraCols / 2);
    padRight = extraCols - padLeft;

    outputImage = zeros(targetHeight, targetWidth, numel(selectedBands), 'single');

    rowStart = padTop + 1;
    rowEnd = rowStart + H - 1;
    colStart = padLeft + 1;
    colEnd = colStart + W - 1;

    outputImage(rowStart:rowEnd, colStart:colEnd, :) = single(normalizedCube);

    % padBottom and padRight are calculated explicitly to document the
    % balanced padding policy and ensure all original pixels are retained.
    assert(padTop + H + padBottom == targetHeight);
    assert(padLeft + W + padRight == targetWidth);
end
