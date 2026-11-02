function [reducedCube, selectedBands, importance, explained] = reduceHypercubePCA(cube, numBands, varianceThreshold)
%REDUCEHYPERCUBEPCA Select original spectral bands using PCA loadings.
%   [reducedCube, selectedBands, importance, explained] = ...
%       reduceHypercubePCA(cube, numBands, varianceThreshold)
%
% INPUTS
%   cube              H-by-W-by-B numeric hyperspectral array.
%   numBands          Number of original spectral bands to retain (default 10).
%   varianceThreshold Cumulative PCA variance to retain in percent (default 95).
%
% OUTPUTS
%   reducedCube       H-by-W-by-numBands cube of original band values.
%   selectedBands     Original band indices, in ascending spectral order.
%   importance        B-by-1 variance-weighted squared PCA loading scores.
%   explained         Percentage of variance explained by each PC.
%
% Requires Statistics and Machine Learning Toolbox (pca).
% Important: For machine learning evaluation, fit PCA/band selection on
% training scans only, then apply the selected bands to validation/test scans.

    if nargin < 2 || isempty(numBands)
        numBands = 10;
    end
    if nargin < 3 || isempty(varianceThreshold)
        varianceThreshold = 95;
    end

    validateattributes(cube, {'numeric'}, {'real','nonempty','3d'}, mfilename, 'cube', 1);
    [H, W, B] = size(cube);
    validateattributes(numBands, {'numeric'}, {'scalar','integer','>=',1,'<=',B}, mfilename, 'numBands', 2);
    validateattributes(varianceThreshold, {'numeric'}, {'scalar','>',0,'<=',100}, mfilename, 'varianceThreshold', 3);

    % One pixel per row; one wavelength/band per column.
    X = reshape(double(cube), [], B);
    validPixels = all(isfinite(X), 2);
    X = X(validPixels, :);
    if size(X, 1) < 2
        error('At least two pixels with finite spectra are required.');
    end

    % Centering is performed internally by pca. No standardization is used.
    [coeff, ~, ~, ~, explained] = pca(X, 'Centered', true, 'Economy', true);
    if isempty(explained) || sum(explained) <= 0
        error('PCA found no nonzero spectral variance.');
    end

    cumulativeVariance = cumsum(explained);
    numPCs = find(cumulativeVariance >= varianceThreshold - 1e-10, 1, 'first');
    if isempty(numPCs)
        numPCs = numel(explained);
    end

    % Squared loadings measure each band's contribution to retained PCs.
    weights = explained(1:numPCs) ./ sum(explained(1:numPCs));
    importance = (coeff(:,1:numPCs).^2) * weights;

    % Sort by importance; then restore spectral order in output cube.
    [~, ranking] = sort(importance, 'descend');
    selectedBands = sort(ranking(1:numBands));
    reducedCube = cube(:,:,selectedBands);

    fprintf('Original cube: %d x %d x %d\n', H, W, B);
    fprintf('PCs explaining %.1f%% variance: %d (actual %.2f%%)\n', ...
        varianceThreshold, numPCs, cumulativeVariance(numPCs));
    fprintf('Selected original bands: %s\n', mat2str(selectedBands(:)'));
    fprintf('Reduced cube: %d x %d x %d\n', size(reducedCube));
end
