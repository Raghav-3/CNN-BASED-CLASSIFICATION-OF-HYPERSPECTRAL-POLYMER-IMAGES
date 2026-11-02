%% select_common_bands_pca.m
% Select 10 common ORIGINAL spectral bands using TRAINING DATA ONLY.
%
% Run after:
%   1) inspect_dataset.m
%   2) create_data_splits.m
%
% Uses the supplied reduceHypercubePCA.m logic:
% variance-weighted squared PCA loadings, retaining PCs up to 95%
% cumulative explained variance.
%
% To keep PCA memory use practical, this script samples up to 500 pixels
% from each training image. The same 10 selected original bands are then
% used for every train, validation, test, and future unseen image.

clc;
close all;

%% 1. Locate project folders and load group-aware split
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);
resultsRoot = fullfile(projectRoot, 'Results');
helperRoot = fullfile(projectRoot, 'HelperCodes');

splitFile = fullfile(resultsRoot, 'dataset_splits.mat');
if ~isfile(splitFile)
    error('Cannot find %s. Run create_data_splits.m first.', splitFile);
end

if ~isfolder(helperRoot)
    error('HelperCodes folder not found: %s', helperRoot);
end

addpath(helperRoot);

S = load(splitFile, 'trainTable', 'validationTable', ...
    'testTable', 'classNames');
trainTable = S.trainTable;
validationTable = S.validationTable;
testTable = S.testTable;
classNames = S.classNames;

%% 2. Settings
numBandsToSelect = 10;
varianceThreshold = 95;  % Retain PCs up to 95% cumulative explained variance.
maxPixelsPerImage = 500;
randomSeed = 2026;

rng(randomSeed, 'twister');

if height(trainTable) == 0
    error('Training table is empty.');
end

% Check that Statistics and Machine Learning Toolbox PCA is available.
if exist('pca', 'file') ~= 2
    error(['The pca function is unavailable. The supplied helper code ' ...
           'requires Statistics and Machine Learning Toolbox.']);
end

%% 3. Sample representative pixel spectra from TRAINING images only
% Each row of X contains one pixel spectrum with 168 band measurements.
% Sampling the same maximum number of pixels per image avoids allowing
% the largest images to dominate the PCA fit.

numTrainingImages = height(trainTable);
numSpectralBands = 168;
numSamplesPerImage = min(maxPixelsPerImage, ...
    min(trainTable.Height .* trainTable.Width));

X = zeros(numTrainingImages * numSamplesPerImage, numSpectralBands, 'double');

fprintf('Sampling training spectra only...\n');
fprintf('Training images: %d\n', numTrainingImages);
fprintf('Maximum sampled pixels per image: %d\n', numSamplesPerImage);

writeRow = 1;

for i = 1:numTrainingImages

    bilPath = char(trainTable.BILPath(i));
    hdrPath = char(trainTable.HDRPath(i));

    % Read each training hyperspectral cube with the required MathWorks API.
    hdr = enviinfo(hdrPath);
    hcube = imhypercube(bilPath);
    cube = gather(hcube);

    [H, W, B] = size(cube);

    if B ~= numSpectralBands || H ~= hdr.Height || ...
            W ~= hdr.Width || B ~= hdr.Bands
        error('HDR/cube dimension mismatch for %s.', bilPath);
    end

    % Convert H-by-W-by-B cube into (H*W)-by-B pixel spectra.
    pixelSpectra = reshape(double(cube), [], B);

    % Remove any pixel spectra containing NaN or Inf values.
    validRows = all(isfinite(pixelSpectra), 2);
    pixelSpectra = pixelSpectra(validRows, :);

    if size(pixelSpectra, 1) < 2
        error('Not enough finite pixel spectra in %s.', bilPath);
    end

    % Randomly sample representative pixels from this training image.
    nAvailable = size(pixelSpectra, 1);
    nTake = min(numSamplesPerImage, nAvailable);
    selectedRows = randperm(nAvailable, nTake);

    sampledSpectra = pixelSpectra(selectedRows, :);

    % Keep the matrix rectangular if an image has fewer valid pixels.
    rowEnd = writeRow + nTake - 1;
    X(writeRow:rowEnd, :) = sampledSpectra;
    writeRow = rowEnd + 1;

    if mod(i, 20) == 0 || i == numTrainingImages
        fprintf('Processed %d / %d training images.\n', i, numTrainingImages);
    end
end

% Remove unused preallocated rows, if any.
X = X(1:writeRow-1, :);

if size(X, 1) < 2
    error('PCA requires at least two valid sampled spectra.');
end

fprintf('\nPCA input matrix: %d pixel spectra x %d bands\n', ...
    size(X, 1), size(X, 2));

%% 4. Fit PCA using training spectra only
% pca centers the variables. We do not z-score/standardize the bands,
% matching the supplied reduceHypercubePCA.m helper.

[coeff, ~, ~, ~, explained] = pca( ...
    X, 'Centered', true, 'Economy', true);

if isempty(explained) || sum(explained) <= 0
    error('PCA found no nonzero spectral variance.');
end

cumulativeVariance = cumsum(explained);
numPCs = find(cumulativeVariance >= varianceThreshold - 1e-10, 1, 'first');

if isempty(numPCs)
    numPCs = numel(explained);
end

actualCumulativeVariance = cumulativeVariance(numPCs);

%% 5. Calculate variance-weighted squared loading importance
% For each original band:
% importance = sum over retained PCs of
%              (squared loading * normalized explained variance).

weights = explained(1:numPCs) ./ sum(explained(1:numPCs));
importance = (coeff(:, 1:numPCs).^2) * weights(:);

% Rank bands by importance, then sort the selected indices into
% original spectral order for consistent downstream image channels.
[~, ranking] = sort(importance, 'descend');
topRankedBands = ranking(1:numBandsToSelect);
selectedBands = sort(topRankedBands(:))';

fprintf('\nPCA selection results:\n');
fprintf('Variance threshold: %.1f%%\n', varianceThreshold);
fprintf('Number of retained PCs: %d\n', numPCs);
fprintf('Actual cumulative explained variance: %.2f%%\n', ...
    actualCumulativeVariance);
fprintf('Selected 10 original band indices: %s\n', ...
    mat2str(selectedBands));

%% 6. Store wavelength information for selected bands
% Wavelength values come from the HDR metadata, if present.
firstTrainingHdr = enviinfo(char(trainTable.HDRPath(1)));

if isfield(firstTrainingHdr, 'Wavelength') && ...
        numel(firstTrainingHdr.Wavelength) == numSpectralBands
    wavelengths = double(firstTrainingHdr.Wavelength(:));
else
    wavelengths = (1:numSpectralBands)';
    warning(['Wavelength metadata unavailable or incomplete. ' ...
             'Band indices will be used instead.']);
end

selectedWavelengths = wavelengths(selectedBands);

%% 7. Save PCA selection outputs
pcaResults = struct();
pcaResults.selectedBands = selectedBands;
pcaResults.topRankedBands = topRankedBands(:)';
pcaResults.importance = importance;
pcaResults.explained = explained;
pcaResults.cumulativeVariance = cumulativeVariance;
pcaResults.numPCs = numPCs;
pcaResults.varianceThreshold = varianceThreshold;
pcaResults.actualCumulativeVariance = actualCumulativeVariance;
pcaResults.wavelengths = wavelengths;
pcaResults.selectedWavelengths = selectedWavelengths;
pcaResults.maxPixelsPerImage = maxPixelsPerImage;
pcaResults.numSampledSpectra = size(X, 1);
pcaResults.randomSeed = randomSeed;

save(fullfile(resultsRoot, 'pca_band_selection.mat'), ...
    'pcaResults', 'selectedBands', 'importance', 'explained', ...
    'cumulativeVariance', 'numPCs', 'wavelengths', ...
    'selectedWavelengths', 'varianceThreshold');

% Save a human-readable table of band importance and selection status.
isSelected = ismember((1:numSpectralBands)', selectedBands(:));
bandTable = table((1:numSpectralBands)', wavelengths(:), ...
    importance(:), isSelected, ...
    'VariableNames', {'BandIndex', 'Wavelength', 'Importance', 'Selected'});

bandTable = sortrows(bandTable, 'Importance', 'descend');
writetable(bandTable, fullfile(resultsRoot, 'pca_band_importance.csv'));

%% 8. Plot cumulative explained variance
fig1 = figure('Name', 'PCA explained variance', 'Color', 'w');
plot(1:numel(cumulativeVariance), cumulativeVariance, ...
    'LineWidth', 1.5);
hold on;
yline(varianceThreshold, '--', ...
    sprintf('%d%% threshold', varianceThreshold));
xline(numPCs, ':', sprintf('%d PCs', numPCs));
grid on;
xlabel('Principal component');
ylabel('Cumulative explained variance (%)');
title('PCA cumulative explained variance (training data only)');
hold off;

exportgraphics(fig1, ...
    fullfile(resultsRoot, 'pca_explained_variance.png'), ...
    'Resolution', 200);

%% 9. Plot importance of all original spectral bands
fig2 = figure('Name', 'Original band importance', 'Color', 'w');
stem(1:numSpectralBands, importance, 'Marker', 'none');
hold on;
stem(selectedBands, importance(selectedBands), ...
    'LineWidth', 1.5, 'Marker', 'o');
grid on;
xlabel('Original spectral band index');
ylabel('Variance-weighted squared loading importance');
title('Original spectral band importance (training data only)');
legend('All bands', 'Selected 10 bands', 'Location', 'best');
hold off;

exportgraphics(fig2, ...
    fullfile(resultsRoot, 'pca_band_importance.png'), ...
    'Resolution', 200);

%% 10. Display selected band information
fprintf('\nSelected band details:\n');
disp(table(selectedBands(:), selectedWavelengths(:), ...
    importance(selectedBands(:)), ...
    'VariableNames', {'BandIndex', 'Wavelength', 'Importance'}));

fprintf('\nCommon PCA band selection completed successfully.\n');
fprintf('The same selectedBands must be applied to train, validation, test, and unseen images.\n');
fprintf('Results saved in: %s\n', resultsRoot);
