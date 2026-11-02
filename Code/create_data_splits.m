%% create_data_splits.m
% Group-aware train/validation/test split for the polymer hyperspectral dataset.
%
% Run after inspect_dataset.m has completed successfully:
%   run('Code/create_data_splits.m')
%
% The supplied dataset has 10 scan groups per class and 5 fragments per
% scan group. Fragments from the same scan are kept in the same subset.
% Overall split: 70% train, 15% validation, 15% test.

clc;

%% 1. Locate project and load inventory
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);
resultsRoot = fullfile(projectRoot, 'Results');

inventoryPath = fullfile(resultsRoot, 'dataset_inventory.mat');
if ~isfile(inventoryPath)
    error('Cannot find %s. Run inspect_dataset.m first.', inventoryPath);
end

loadedData = load(inventoryPath, 'inventory', 'classNames');
inventory = loadedData.inventory;
classNames = loadedData.classNames;

requiredVariables = {'Class', 'ScanID', 'FileName', 'BILPath', 'HDRPath'};
if ~all(ismember(requiredVariables, inventory.Properties.VariableNames))
    error('Dataset inventory is missing one or more required columns.');
end

if any(inventory.ScanID == "UNKNOWN")
    error('Some scan IDs could not be extracted. Inspect the filenames before splitting.');
end

%% 2. Check scan-group structure
rng(42, 'twister');  % Fixed seed for reproducibility.

numClasses = numel(classNames);
groupLists = cell(numClasses, 1);
groupSizes = cell(numClasses, 1);

fprintf('Checking scan groups...\n');

for c = 1:numClasses
    classRows = inventory(inventory.Class == classNames(c), :);
    groups = unique(classRows.ScanID, 'stable');

    groupLists{c} = groups;
    groupSizes{c} = zeros(numel(groups), 1);

    for g = 1:numel(groups)
        groupSizes{c}(g) = sum(classRows.ScanID == groups(g));
    end

    fprintf('%s: %d images, %d scan groups; group sizes: %s\n', ...
        classNames(c), height(classRows), numel(groups), ...
        mat2str(groupSizes{c}'));

    if numel(groups) ~= 10
        error(['Expected 10 scan groups in class %s, but found %d. ' ...
               'Review the grouping rule before splitting.'], ...
               classNames(c), numel(groups));
    end

    if any(groupSizes{c} ~= 5)
        error(['Expected 5 fragments per scan group in class %s. ' ...
               'Observed group sizes: %s. Review grouping before splitting.'], ...
               classNames(c), mat2str(groupSizes{c}'));
    end
end

%% 3. Assign complete scan groups to subsets
% Each class has 10 groups:
%   7 groups -> training (70%)
%   remaining 3 groups -> validation/test.
%
% To achieve exactly 15% validation and 15% test overall, two classes
% contribute 1 validation + 2 test groups, and two classes contribute
% 2 validation + 1 test groups. All classes remain represented in all sets.

splitLabel = strings(height(inventory), 1);
groupAssignment = table();

for c = 1:numClasses
    classRows = inventory.Class == classNames(c);
    groups = groupLists{c};

    % Shuffle groups reproducibly within each class.
    groups = groups(randperm(numel(groups)));

    if c <= 2
        numValidationGroups = 1;
        numTestGroups = 2;
    else
        numValidationGroups = 2;
        numTestGroups = 1;
    end

    trainGroups = groups(1:7);
    validationGroups = groups(8:7 + numValidationGroups);
    testGroups = groups(8 + numValidationGroups:end);

    for g = 1:numel(trainGroups)
        rows = classRows & inventory.ScanID == trainGroups(g);
        splitLabel(rows) = "train";
    end

    for g = 1:numel(validationGroups)
        rows = classRows & inventory.ScanID == validationGroups(g);
        splitLabel(rows) = "validation";
    end

    for g = 1:numel(testGroups)
        rows = classRows & inventory.ScanID == testGroups(g);
        splitLabel(rows) = "test";
    end

    thisClassAssignment = table( ...
        repmat(classNames(c), 10, 1), groups(:), ...
        [repmat("train", 7, 1); ...
         repmat("validation", numValidationGroups, 1); ...
         repmat("test", numTestGroups, 1)], ...
        'VariableNames', {'Class', 'ScanID', 'Subset'});

    groupAssignment = [groupAssignment; thisClassAssignment]; %#ok<AGROW>
end

if any(splitLabel == "")
    error('Some images were not assigned to a subset.');
end

%% 4. Add subset labels and verify no scan leakage
splitTable = inventory;
splitTable.Subset = splitLabel;

% Every Class + ScanID combination must belong to exactly one subset.
[groupID, groupClass, groupScan] = findgroups( ...
    splitTable.Class, splitTable.ScanID); %#ok<ASGLU>

for g = 1:max(groupID)
    labelsForGroup = unique(splitTable.Subset(groupID == g));
    if numel(labelsForGroup) ~= 1
        error('Data leakage detected: one scan group appears in multiple subsets.');
    end
end

%% 5. Print subset counts
fprintf('\nOverall subset counts:\n');
subsetNames = ["train"; "validation"; "test"];
for s = 1:numel(subsetNames)
    count = sum(splitTable.Subset == subsetNames(s));
    fprintf('%-12s %3d images (%.1f%%)\n', ...
        subsetNames(s), count, 100 * count / height(splitTable));
end

fprintf('\nClass-wise subset counts:\n');
for c = 1:numClasses
    fprintf('\n%s\n', classNames(c));
    for s = 1:numel(subsetNames)
        count = sum(splitTable.Class == classNames(c) & ...
                    splitTable.Subset == subsetNames(s));
        fprintf('  %-12s %2d images\n', subsetNames(s), count);
    end
end

%% 6. Save split files
writetable(splitTable, fullfile(resultsRoot, 'dataset_splits.csv'));
writetable(groupAssignment, fullfile(resultsRoot, 'scan_group_assignments.csv'));

trainTable = splitTable(splitTable.Subset == "train", :);
validationTable = splitTable(splitTable.Subset == "validation", :);
testTable = splitTable(splitTable.Subset == "test", :);

save(fullfile(resultsRoot, 'dataset_splits.mat'), ...
    'splitTable', 'groupAssignment', 'trainTable', ...
    'validationTable', 'testTable', 'classNames');

fprintf('\nGroup-aware splitting completed successfully.\n');
fprintf('Saved split files in: %s\n', resultsRoot);
