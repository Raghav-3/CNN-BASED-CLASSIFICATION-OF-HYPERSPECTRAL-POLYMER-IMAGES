%% evaluate_all_experiments.m
% Evaluate all three trained CNN models on the same held-out test set.
% MATLAB R2026b | CPU. This script does not train or modify any model.

clear; clc; close all;

%% 1. Locate project folders
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);
resultsFolder = fullfile(projectRoot, 'Results');
modelFolder = fullfile(projectRoot, 'Model');

datastoreFile = fullfile(resultsFolder, 'prepared_datastores.mat');
if ~isfile(datastoreFile)
    error('Prepared datastores not found: %s', datastoreFile);
end

modelFiles = {
    fullfile(modelFolder, 'baseline_cnn.mat')
    fullfile(modelFolder, 'experiment2_dropout_cnn.mat')
    fullfile(modelFolder, 'experiment3_shallow_cnn.mat')
};
modelNames = {'Baseline'; 'Dropout CNN'; 'Shallow CNN'};

for k = 1:numel(modelFiles)
    if ~isfile(modelFiles{k})
        error('Model file not found: %s', modelFiles{k});
    end
end

%% 2. Load the held-out test datastore and saved models
D = load(datastoreFile, 'imdsTest');
imdsTest = D.imdsTest;

B = load(modelFiles{1}, 'net', 'classNames', 'inputSize');
E2 = load(modelFiles{2}, 'netExperiment2', 'classNames', 'inputSize');
E3 = load(modelFiles{3}, 'netExperiment3', 'classNames', 'inputSize');

nets = {B.net; E2.netExperiment2; E3.netExperiment3};
classNames = B.classNames;
inputSize = B.inputSize;

if ~isequal(E2.classNames, classNames) || ~isequal(E3.classNames, classNames)
    error('The three models do not have identical class ordering.');
end
if ~isequal(E2.inputSize, inputSize) || ~isequal(E3.inputSize, inputSize)
    error('The three models do not have identical input dimensions.');
end
if ~isequal(inputSize, [68 73 10])
    error('Unexpected model input size: [%s].', num2str(inputSize));
end

fprintf('Test images: %d\n', numel(imdsTest.Files));
fprintf('Classes: %s\n', strjoin(classNames, ', '));
fprintf('Input size: %d x %d x %d\n', inputSize);

%% 3. Prepare the test set once (no augmentation)
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest);
YTrue = categorical(string(imdsTest.Labels), classNames);

%% 4. Evaluate all three models
allSummary = table();
allClassMetrics = table();
allConfusionMatrices = cell(numel(nets), 1);
allPredictions = cell(numel(nets), 1);

for modelIndex = 1:numel(nets)
    fprintf('\n========================================\n');
    fprintf('Evaluating %s\n', modelNames{modelIndex});
    fprintf('========================================\n');

    scores = minibatchpredict(nets{modelIndex}, augTest, ...
        'MiniBatchSize', 16, ...
        'ExecutionEnvironment', 'cpu');

    YPred = scores2label(scores, categorical(classNames));
    YPred = categorical(string(YPred), classNames);

    % Rows are true classes; columns are predicted classes.
    C = confusionmat(YTrue, YPred, 'Order', categorical(classNames));

    actualCount = sum(C, 2);
    predictedCount = sum(C, 1)';
    truePositive = diag(C);

    precision = truePositive ./ max(predictedCount, 1);
    recall = truePositive ./ max(actualCount, 1);
    f1 = zeros(size(precision));
    validF1 = (precision + recall) > 0;
    f1(validF1) = 2 .* precision(validF1) .* recall(validF1) ./ ...
        (precision(validF1) + recall(validF1));

    accuracy = sum(truePositive) / sum(C(:));
    macroF1 = mean(f1);
    classWeights = actualCount / sum(actualCount);
    weightedF1 = sum(classWeights .* f1);

    summaryRow = table(string(modelNames{modelIndex}), accuracy, ...
        macroF1, weightedF1, ...
        'VariableNames', {'Model', 'Accuracy', 'MacroF1', 'WeightedF1'});
    allSummary = [allSummary; summaryRow]; %#ok<AGROW>

    classTable = table(repmat(string(modelNames{modelIndex}), numel(classNames), 1), ...
        string(classNames(:)), actualCount, predictedCount, ...
        precision, recall, f1, ...
        'VariableNames', {'Model', 'Class', 'ActualCount', ...
        'PredictedCount', 'Precision', 'Recall', 'F1Score'});
    allClassMetrics = [allClassMetrics; classTable]; %#ok<AGROW>

    allConfusionMatrices{modelIndex} = C;
    allPredictions{modelIndex} = YPred;

    fprintf('Accuracy    : %.2f%%\n', 100 * accuracy);
    fprintf('Macro F1    : %.4f\n', macroF1);
    fprintf('Weighted F1 : %.4f\n', weightedF1);
    disp(classTable(:, {'Class','ActualCount','PredictedCount', ...
        'Precision','Recall','F1Score'}));

    fig = figure('Name', [modelNames{modelIndex} ' Test Confusion Matrix']);
    confusionchart(C, classNames, ...
        'Title', [modelNames{modelIndex} ' - Test Confusion Matrix'], ...
        'RowSummary', 'row-normalized', ...
        'ColumnSummary', 'column-normalized');

    safeName = lower(strrep(modelNames{modelIndex}, ' ', '_'));
    exportgraphics(fig, fullfile(resultsFolder, ...
        [safeName '_test_confusion_matrix.png']));
end

%% 5. Display and save comparison tables
fprintf('\n========== MODEL COMPARISON: TEST SET ==========\n');
disp(allSummary);
fprintf('\n========== CLASS-WISE TEST METRICS ==========\n');
disp(allClassMetrics);

writetable(allSummary, fullfile(resultsFolder, ...
    'all_models_test_summary.csv'));
writetable(allClassMetrics, fullfile(resultsFolder, ...
    'all_models_test_class_metrics.csv'));

%% 6. Plot accuracy and F1 comparison
metricValues = [allSummary.Accuracy, allSummary.MacroF1, ...
    allSummary.WeightedF1];

figComparison = figure('Name', 'All CNN Experiments Test Comparison');
bar(metricValues);
xticks(1:height(allSummary));
xticklabels(cellstr(allSummary.Model));
xtickangle(15);
ylabel('Score');
ylim([0 1.05]);
title('Test Performance Comparison of CNN Experiments');
legend({'Accuracy', 'Macro F1', 'Weighted F1'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');
grid on;

exportgraphics(figComparison, fullfile(resultsFolder, ...
    'all_models_test_comparison.png'));

%% 7. Save complete evaluation results
save(fullfile(resultsFolder, 'all_models_test_evaluation.mat'), ...
    'allSummary', 'allClassMetrics', 'allConfusionMatrices', ...
    'allPredictions', 'classNames', 'modelNames');

fprintf('\nAll model test evaluations saved in:\n%s\n', resultsFolder);
fprintf('EVALUATION COMPLETE.\n');
fprintf(['These test results are for final reporting only. ' ...
    'Do not tune models using this test set.\n']);
