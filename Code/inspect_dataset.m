%% inspect_dataset.m
% Step 1: Inspect all hyperspectral BIL/HDR images.
% Requires the Hyperspectral Imaging Library for Image Processing Toolbox.
%
% IMPORTANT:
% Save this file in the project's Code folder:
% Polymer_CNN_Project/Code/inspect_dataset.m
%
% Then run:
% run('Code/inspect_dataset.m')

clc;
close all;

%% 1. Locate project folders
% Derive the project root from this script's location, not from pwd.
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

datasetRoot = fullfile(projectRoot, 'Dataset');
outputRoot  = fullfile(projectRoot, 'Results');

if ~isfolder(datasetRoot)
    error(['Dataset folder not found:\n%s\n' ...
           'Expected Dataset to be alongside Code, not inside Code.'], datasetRoot);
end

if ~isfolder(outputRoot)
    mkdir(outputRoot);
end

classNames = ["HDPE", "PET", "PP", "PS"];

% These bands are used only to create a pseudo-RGB visualization.
% They are NOT the PCA-selected bands used later for CNN training.
pseudoRGBBands = [30, 20, 10];  % [Red, Green, Blue]

%% 2. Find all BIL files recursively
bilFiles = dir(fullfile(datasetRoot, '**', '*.bil'));

if isempty(bilFiles)
    error('No .bil files were found under: %s', datasetRoot);
end

fprintf('Found %d BIL files.\n', numel(bilFiles));

%% 3. Prepare arrays for the dataset inventory
numFiles = numel(bilFiles);

filePathColumn = strings(numFiles, 1);
hdrPathColumn  = strings(numFiles, 1);
classColumn    = strings(numFiles, 1);
fileNameColumn = strings(numFiles, 1);
scanIDColumn   = strings(numFiles, 1);

rowsColumn  = zeros(numFiles, 1);
colsColumn  = zeros(numFiles, 1);
bandsColumn = zeros(numFiles, 1);

% Store one representative cube from each class for plotting.
representativeCubes = cell(numel(classNames), 1);
representativeNames = strings(numel(classNames), 1);

%% 4. Read and validate every BIL/HDR pair
for k = 1:numFiles

    % Full path to the BIL file.
    bilPath = fullfile(bilFiles(k).folder, bilFiles(k).name);

    % Extract filename without extension, then construct matching HDR path.
    [~, baseName] = fileparts(bilFiles(k).name);
    hdrPath = fullfile(bilFiles(k).folder, [baseName, '.hdr']);

    if ~isfile(hdrPath)
        error('Matching HDR file not found:\n%s', hdrPath);
    end

    % Determine class label from the parent folder.
    [~, classFolder] = fileparts(bilFiles(k).folder);
    classLabel = string(classFolder);

    if ~any(classNames == classLabel)
        error('Unexpected class folder "%s" for file:\n%s', classLabel, bilPath);
    end

    % Read HDR metadata and load the hyperspectral image.
    hdr = enviinfo(hdrPath);
    hcube = imhypercube(bilPath);
    cube = gather(hcube);

    % Confirm that the imported cube is three-dimensional.
    if ndims(cube) ~= 3
        error('Expected a 3-D cube, but received %d dimensions:\n%s', ...
            ndims(cube), bilPath);
    end

    [H, W, B] = size(cube);

    % Verify imported dimensions against HDR metadata.
    if H ~= hdr.Height || W ~= hdr.Width || B ~= hdr.Bands
        error(['Dimension mismatch for:\n%s\n' ...
               'Cube size: %d x %d x %d\n' ...
               'HDR size : %d x %d x %d'], ...
               bilPath, H, W, B, hdr.Height, hdr.Width, hdr.Bands);
    end

    % This project expects 168 spectral bands in every image.
    if B ~= 168
        error('Expected 168 spectral bands, but found %d in:\n%s', B, bilPath);
    end

    % Store metadata for the inventory table.
    filePathColumn(k) = string(bilPath);
    hdrPathColumn(k)  = string(hdrPath);
    classColumn(k)    = classLabel;
    fileNameColumn(k) = string(bilFiles(k).name);

    rowsColumn(k)  = H;
    colsColumn(k)  = W;
    bandsColumn(k) = B;

    % Extract scan ID from filenames such as scan0001_frag02.
    scanToken = regexp(baseName, 'scan\d+', 'match', 'once');

    if isempty(scanToken)
        scanIDColumn(k) = "UNKNOWN";
    else
        scanIDColumn(k) = string(scanToken);
    end

    % Keep the first image encountered for each class for visualization.
    classIndex = find(classNames == classLabel, 1);

    if isempty(representativeCubes{classIndex})
        representativeCubes{classIndex} = cube;
        representativeNames(classIndex) = string(bilFiles(k).name);
    end

    if mod(k, 20) == 0 || k == numFiles
        fprintf('Read %d / %d images.\n', k, numFiles);
    end
end

%% 5. Create and save dataset inventory
inventory = table( ...
    classColumn, fileNameColumn, scanIDColumn, ...
    rowsColumn, colsColumn, bandsColumn, ...
    filePathColumn, hdrPathColumn, ...
    'VariableNames', { ...
        'Class', 'FileName', 'ScanID', ...
        'Height', 'Width', 'NumBands', ...
        'BILPath', 'HDRPath'});

fprintf('\nImage counts by class:\n');
classCounts = groupcounts(inventory, 'Class');
disp(classCounts);

fprintf('Height range: %d to %d pixels\n', min(rowsColumn), max(rowsColumn));
fprintf('Width range : %d to %d pixels\n', min(colsColumn), max(colsColumn));
fprintf('Band counts : %s\n', mat2str(unique(bandsColumn)'));

writetable(inventory, fullfile(outputRoot, 'dataset_inventory.csv'));

save(fullfile(outputRoot, 'dataset_inventory.mat'), ...
    'inventory', 'classNames', 'pseudoRGBBands');

%% 6. Display one pseudo-RGB image per polymer class
figure('Name', 'Representative hyperspectral images', 'Color', 'w');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for c = 1:numel(classNames)

    cube = double(representativeCubes{c});

    % Map three selected spectral bands to display RGB channels.
    rgb = zeros(size(cube, 1), size(cube, 2), 3);

    for ch = 1:3
        bandImage = cube(:, :, pseudoRGBBands(ch));

        bandMin = min(bandImage(:));
        bandMax = max(bandImage(:));

        % Normalize this channel for display only.
        if bandMax > bandMin
            rgb(:, :, ch) = (bandImage - bandMin) ./ (bandMax - bandMin);
        end
    end

    nexttile;
    imshow(rgb);
    title(classNames(c) + " - " + representativeNames(c), ...
        'Interpreter', 'none');
end

exportgraphics(gcf, ...
    fullfile(outputRoot, 'representative_images.png'), ...
    'Resolution', 200);

%% 7. Plot one representative center-pixel spectrum per class
figure('Name', 'Representative pixel spectra', 'Color', 'w');
hold on;
grid on;

for c = 1:numel(classNames)

    cube = double(representativeCubes{c});

    centerRow = round(size(cube, 1) / 2);
    centerCol = round(size(cube, 2) / 2);

    spectrum = squeeze(cube(centerRow, centerCol, :));

    plot(1:numel(spectrum), spectrum, ...
        'LineWidth', 1.2, ...
        'DisplayName', classNames(c));
end

xlabel('Original spectral band index');
ylabel('Recorded pixel value');
title('Representative center-pixel spectral signatures');
legend('Location', 'best');
hold off;

exportgraphics(gcf, ...
    fullfile(outputRoot, 'representative_spectra.png'), ...
    'Resolution', 200);

%% 8. Finish
fprintf('\nStep 1 completed successfully.\n');
fprintf('Inventory and figures saved in:\n%s\n', outputRoot);
