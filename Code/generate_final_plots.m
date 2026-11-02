%% generate_final_plots.m
% Creates report-ready plots from the saved CNN training/evaluation results.
% Run from the Polymer_CNN_Project root folder.

%% generate_final_plots.m
clear; close all; clc;

% Automatically identify the project root from this script's location
scriptPath = mfilename('fullpath');
codeDir = fileparts(scriptPath);
projectRoot = fileparts(codeDir);

resultsDir = fullfile(projectRoot, 'Results');
plotsDir = fullfile(resultsDir, 'Plots');

if ~isfolder(resultsDir)
    error('Results folder not found at: %s', resultsDir);
end

if ~isfolder(plotsDir)
    mkdir(plotsDir);
end

%% 1. Load the three saved training histories
S1 = load(fullfile(resultsDir, 'baseline_training_info.mat'));
S2 = load(fullfile(resultsDir, 'experiment2_dropout_training_info.mat'));
S3 = load(fullfile(resultsDir, 'experiment3_shallow_training_info.mat'));

history = cell(3,1);
history{1} = S1.trainingInfo;
history{2} = S2.trainingInfoExperiment2;
history{3} = S3.trainingInfoExperiment3;

modelNames = {'Baseline CNN', 'Dropout CNN', 'Shallow CNN'};

% Check that the expected columns exist before plotting.
for k = 1:3
    trainTable = history{k}.TrainingHistory;
    valTable = history{k}.ValidationHistory;

    requiredTrain = {'Iteration','Loss'};
    requiredVal = {'Iteration','Loss'};

    if ~all(ismember(requiredTrain, trainTable.Properties.VariableNames)) || ...
       ~all(ismember(requiredVal, valTable.Properties.VariableNames))
        error('Unexpected training-history table format for %s.', modelNames{k});
    end
end

%% 2. Training and validation loss curves
fig1 = figure('Color','w','Position',[100 100 1050 850]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

for k = 1:3
    trainTable = history{k}.TrainingHistory;
    valTable = history{k}.ValidationHistory;

    nexttile;
    plot(trainTable.Iteration, trainTable.Loss, ...
        'LineWidth',1.0,'DisplayName','Training loss');
    hold on;
    plot(valTable.Iteration, valTable.Loss, 'o-', ...
        'LineWidth',1.5,'MarkerSize',4,'DisplayName','Validation loss');

    grid on;
    xlabel('Iteration');
    ylabel('Cross-entropy loss');
    title(modelNames{k});
    legend('Location','northeast');
end

sgtitle('Training and Validation Loss: CNN Experiments');
exportgraphics(fig1,fullfile(plotsDir,'training_curves.png'),'Resolution',300);

%% 3. Class-wise precision, recall and F1-score
classFile = fullfile(resultsDir,'all_models_test_class_metrics.csv');
if ~isfile(classFile)
    error('Missing file: %s',classFile);
end

classMetrics = readtable(classFile);
needed = {'Model','Class','Precision','Recall','F1Score'};
if ~all(ismember(needed,classMetrics.Properties.VariableNames))
    error('Class metrics CSV does not contain the expected columns.');
end

modelColumn = string(classMetrics.Model);
classColumn = string(classMetrics.Class);
classNames = {'HDPE','PET','PP','PS'};
modelLabels = {'Baseline','Dropout CNN','Shallow CNN'};

fig2 = figure('Color','w','Position',[100 100 1350 430]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

for m = 1:3
    nexttile;
    precisionValues = nan(1,4);
    recallValues = nan(1,4);
    f1Values = nan(1,4);

    for c = 1:4
        idx = modelColumn == modelLabels{m} & classColumn == classNames{c};
        if nnz(idx) ~= 1
            error('Expected exactly one row for %s / %s in class metrics CSV.', ...
                modelLabels{m},classNames{c});
        end
        precisionValues(c) = classMetrics.Precision(idx);
        recallValues(c) = classMetrics.Recall(idx);
        f1Values(c) = classMetrics.F1Score(idx);
    end

    values = [precisionValues(:),recallValues(:),f1Values(:)];
    bar(values);
    ylim([0 1.05]);
    xticks(1:4);
    xticklabels(classNames);
    ylabel('Score');
    title(modelLabels{m});
    grid on;
    if m == 1
        legend({'Precision','Recall','F1-score'},'Location','southoutside');
    end
end

sgtitle('Class-wise Test Metrics');
exportgraphics(fig2,fullfile(plotsDir,'classwise_metrics.png'),'Resolution',300);

%% 4. Experiment-level comparison including mean recall
summaryFile = fullfile(resultsDir,'all_models_test_summary.csv');
if ~isfile(summaryFile)
    error('Missing file: %s',summaryFile);
end

summaryTable = readtable(summaryFile);
summaryModels = string(summaryTable.Model);

accuracy = nan(3,1);
macroPrecision = nan(3,1);
meanRecall = nan(3,1);
macroF1 = nan(3,1);
weightedF1 = nan(3,1);

for m = 1:3
    idxSummary = summaryModels == modelLabels{m};
    if nnz(idxSummary) ~= 1
        error('Expected exactly one summary row for %s.',modelLabels{m});
    end

    accuracy(m) = summaryTable.Accuracy(idxSummary);
    macroF1(m) = summaryTable.MacroF1(idxSummary);
    weightedF1(m) = summaryTable.WeightedF1(idxSummary);

    idxClass = modelColumn == modelLabels{m};
    macroPrecision(m) = mean(classMetrics.Precision(idxClass),'omitnan');
    meanRecall(m) = mean(classMetrics.Recall(idxClass),'omitnan');
end

comparisonTable = table(string(modelLabels(:)),accuracy,meanRecall, ...
    macroPrecision,macroF1,weightedF1, ...
    'VariableNames',{'Model','Accuracy','MeanRecall','MacroPrecision', ...
    'MacroF1','WeightedF1'});

writetable(comparisonTable,fullfile(resultsDir,'final_experiment_metrics.csv'));

fig3 = figure('Color','w','Position',[100 100 1050 550]);
bar([accuracy,meanRecall,macroPrecision,macroF1,weightedF1]);
ylim([0 1.05]);
xticks(1:3);
xticklabels(modelLabels);
ylabel('Metric value');
title('Overall Test Metric Comparison');
legend({'Accuracy','Mean recall','Macro precision','Macro F1','Weighted F1'}, ...
    'Location','southoutside','NumColumns',3);
grid on;
exportgraphics(fig3,fullfile(plotsDir,'experiment_comparison.png'),'Resolution',300);

%% 5. Print a compact summary
fprintf('\nFinal experiment metrics:\n');
disp(comparisonTable);

fprintf('\nPlots saved in: %s\n',plotsDir);
fprintf(' - training_curves.png\n');
fprintf(' - classwise_metrics.png\n');
fprintf(' - experiment_comparison.png\n');
fprintf('Metrics table saved as Results/final_experiment_metrics.csv\n');
