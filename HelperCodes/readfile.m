clc; clear; close all;

% Specify file paths
hdrFile = '.\PET\scan0001_frag01.hdr';
bilFile = '.\PET\scan0001_frag01.bil';

% Read header file
hdr = enviinfo(hdrFile);

% Read hyperspectral cube
hcube = imhypercube(bilFile);

% this is the hyperspectral data 38X39X168 here 168 represents number of bands
cube = gather(hcube);



% Check dimensions
[rows, cols, bands] = size(cube);

fprintf('Rows: %d\n', rows);
fprintf('Columns: %d\n', cols);
fprintf('Spectral bands: %d\n', bands);

%pseudo RGB
rBand = 30; gBand = 20; bBand = 10;   % choose appropriate band numbers
R = cube(:,:,rBand);
G = cube(:,:,gBand);
B = cube(:,:,bBand);

% scale each channel to [0,1]
R = mat2gray(R);
G = mat2gray(G);
B = mat2gray(B);

rgb = cat(3,R,G,B);
imshow(rgb);

numBands = 10;          % Retain 10 original spectral bands
varianceThreshold = 95; % Use PCs explaining 95% variance

[reducedCube, selectedBands, importance, explained] = ...
    reduceHypercubePCA(cube, numBands, varianceThreshold);


%pseudo RGB
rBand = 2; gBand = 5; bBand = 10;   % choose appropriate band numbers
R = reducedCube(:,:,rBand);
G = reducedCube(:,:,gBand);
B = reducedCube(:,:,bBand);

% scale each channel to [0,1]
R = mat2gray(R);
G = mat2gray(G);
B = mat2gray(B);

rgb = cat(3,R,G,B);
figure
imshow(rgb);