%% evaluate_cnn.m
% Independent test-set evaluation of the trained baseline CNN.
% MATLAB R2026b | CPU
%
% Uses the 30 test images only after training and model selection are complete.

clear; clc; close all;

%% 1. Locate project files
scriptFolder = fileparts(mfilename('fullpath'));
projectRoot = fileparts(scriptFolder);

resultsFolder = fullfile(projectRoot, 'Results');
modelFolder = fullfile(projectRoot, 'Model');

modelFile = fullfile(modelFolder, 'baseline_cnn.mat');
datastoreFile = fullfile(resultsFolder, 'prepared_datastores.mat');

if ~isfile(modelFile)
    error('Trained model not found: %s', modelFile);
end
if ~isfile(datastoreFile)
    error('Prepared datastores not found: %s', datastoreFile);
end

%% 2. Load model and test datastore
M = load(modelFile, 'net', 'classNames', 'inputSize');
D = load(datastoreFile, 'imdsTest');

net = M.net;
classNames = M.classNames;
inputSize = M.inputSize;
imdsTest = D.imdsTest;

if numel(imdsTest.Files) ~= 30
    warning('Expected 30 test images, but found %d.', numel(imdsTest.Files));
end

fprintf('Test images: %d\n', numel(imdsTest.Files));
fprintf('Classes: %s\n', strjoin(classNames, ', '));

%% 3. Prepare test images
% No augmentation is applied to the test set.
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%% 4. Predict test-set class scores
fprintf('\nPredicting test-set labels on CPU...\n');
scores = minibatchpredict(net, augTest, ...
    'MiniBatchSize', 16, ...
    'ExecutionEnvironment', 'cpu');

% Convert network scores into categorical class predictions.
classNamesCategorical = categorical(classNames);
YPred = scores2label(scores, classNamesCategorical);
YTrue = imdsTest.Labels;

% Ensure both vectors use the same category order.
YTrue = categorical(string(YTrue), classNames);
YPred = categorical(string(YPred), classNames);

%% 5. Calculate confusion matrix and metrics
C = confusionmat(YTrue, YPred, 'Order', categorical(classNames));

truePerClass = sum(C, 2);
predictedPerClass = sum(C, 1)';

truePositive = diag(C);
precision = truePositive ./ max(predictedPerClass, 1);
recall = truePositive ./ max(truePerClass, 1);
f1 = 2 .* precision .* recall ./ max(precision + recall, eps);

% If a class has no predicted samples, its precision and F1 are set to zero.
precision(predictedPerClass == 0) = 0;
f1((precision + recall) == 0) = 0;

accuracy = sum(truePositive) / sum(C(:));
macroF1 = mean(f1);
classWeights = truePerClass / sum(truePerClass);
weightedF1 = sum(classWeights .* f1);

fprintf('\n========== TEST RESULTS ==========\n');
fprintf('Test accuracy : %.2f%%\n', 100 * accuracy);
fprintf('Macro F1      : %.4f\n', macroF1);
fprintf('Weighted F1   : %.4f\n', weightedF1);

metricsTable = table(classNames(:), truePerClass, predictedPerClass, ...
    precision, recall, f1, ...
    'VariableNames', {'Class', 'ActualCount', 'PredictedCount', ...
    'Precision', 'Recall', 'F1Score'});

disp(metricsTable);

%% 6. Display and save confusion matrix
figure('Name', 'Baseline CNN Test Confusion Matrix');
confusionchart(C, classNames, ...
    'Title', 'Baseline CNN - Test Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

exportgraphics(gcf, fullfile(resultsFolder, 'baseline_test_confusion_matrix.png'));

%% 7. Save test metrics
summaryTable = table(accuracy, macroF1, weightedF1, ...
    'VariableNames', {'Accuracy', 'MacroF1', 'WeightedF1'});

writetable(metricsTable, fullfile(resultsFolder, 'baseline_test_class_metrics.csv'));
writetable(summaryTable, fullfile(resultsFolder, 'baseline_test_summary.csv'));

save(fullfile(resultsFolder, 'baseline_test_results.mat'), ...
    'C', 'metricsTable', 'summaryTable', 'YPred', 'YTrue', 'scores', 'classNames');

fprintf('\nSaved test results in:\n%s\n', resultsFolder);
fprintf('STEP 6 COMPLETE: Independent test evaluation finished.\n');
