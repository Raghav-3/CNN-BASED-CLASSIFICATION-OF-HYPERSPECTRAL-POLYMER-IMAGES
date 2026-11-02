# CNN-Based Classification of Hyperspectral Polymer Images

**Course:** ME3101 – Autumn 2026  
**Project:** Mini-Project 2

## 1. Project overview

This project classifies hyperspectral images into four polymer classes: HDPE, PET, PP and PS. The dataset contains BIL images with matching HDR metadata and 168 spectral bands. The workflow reads the hyperspectral cubes, selects one common set of 10 original bands using PCA fitted on training data, applies training-only normalization, pads images to a common spatial size, and trains 2D CNN models.

## 2. MATLAB requirements

Developed and tested with MATLAB R2026b. Required toolboxes:

- Image Processing Toolbox
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox

The project was trained on CPU; a GPU is not required.

## 3. Folder structure

```text
Polymer_CNN_Project/
├── Dataset/
│   ├── HDPE/
│   ├── PET/
│   ├── PP/
│   └── PS/
├── HelperCodes/
│   ├── readfile.m
│   └── reduceHypercubePCA.m
├── Code/
├── Model/
├── Results/
│   └── Plots/
└── Report/
```

Keep the four class folders and their BIL/HDR pairs in `Dataset`. The dataset may be supplied separately; do not add an unnecessary duplicate to the final ZIP.

## 4. Execution order

Open MATLAB and change to the project root:

```matlab
cd('C:\path\to\Polymer_CNN_Project')
```

Run these scripts in order to reproduce the workflow from the supplied dataset:

1. `Code/inspect_dataset.m` — reads BIL/HDR files, checks dimensions and band counts, and creates the inventory and exploratory plots.
2. `Code/create_data_splits.m` — creates fixed train, validation and test partitions while keeping related scan groups together.
3. `Code/select_common_bands_pca.m` — fits PCA using training spectra and selects the common 10 original bands.
4. `Code/prepare_datastores.m` — calculates training-only normalization statistics, applies the selected bands, pads images and creates datastores.
5. `Code/train_cnn.m` — trains the baseline CNN.
6. `Code/experiment2_dropout_cnn.m` — trains the dropout experiment.
7. `Code/experiment3_shallow_cnn.m` — trains the shallow CNN.
8. `Code/evaluate_cnn.m` and `Code/evaluate_all_experiments.m` — evaluate the models and save metrics/confusion matrices.
9. `Code/generate_final_plots.m` — generates comparison plots from saved histories and evaluation tables.
10. `Code/predict_new_polymer.m` — predicts the class of a selected BIL/HDR image.

**Warning:** Training scripts retrain the networks and may overwrite saved models/results. For inference only, do not rerun PCA, normalization fitting or training. Use the saved model and preprocessing configuration.

## 5. Preprocessing

- Original images contain 168 spectral bands.
- One common set of 10 original band indices is selected using training data only.
- The same band indices are applied to training, validation, test and inference images.
- Per-band normalization statistics are calculated from training images only.
- Images are padded symmetrically to `68 × 73 × 10`.
- Original pixels are retained; images are not cropped or resized.
- During inference, images exceeding the configured target dimensions are rejected rather than cropped.

The selected bands, PCA information, normalization parameters, target dimensions, padding settings and class names are saved in the preprocessing MAT files. Use these saved settings during inference; do not refit PCA or normalization on a new image.

## 6. CNN experiments

Three configurations were evaluated using the same data partitions:

1. **Baseline CNN:** three convolution blocks, batch normalization, ReLU activations, downsampling, global average pooling and a four-class classification head.
2. **Dropout CNN:** baseline-style architecture with dropout probability 0.30.
3. **Shallow CNN:** two convolution blocks instead of three.

The baseline had the lowest validation loss among the three experiments. All three models achieved 100% accuracy, macro F1 and weighted F1 on the 30-image test set. The baseline is selected based on its validation performance.

## 7. Evaluation and limitations

The evaluation outputs include overall accuracy, class-wise precision/recall/F1, macro F1, weighted F1, confusion matrices, model comparison tables and training/validation loss curves.

The test set contains 30 images. The reported 100% accuracy describes performance on this particular held-out test set; it does not guarantee performance on new specimens or acquisition conditions. The data were split by scan group to reduce leakage, but the limited test-set size and possible acquisition-related similarities should be acknowledged in the report.

The prediction demonstration performed so far used an image from the test set. Describe it as a **held-out test-image inference demonstration**, not an independent external unseen-specimen evaluation. If an additional independent BIL/HDR image becomes available, use it for a stronger unseen-image demonstration.

## 8. Prediction

Run:

```matlab
run('Code/predict_new_polymer.m')
```

Select a `.bil` file when prompted. The script finds the corresponding `.hdr` file, applies the saved band selection, normalization and padding, and predicts HDPE, PET, PP or PS using the saved baseline network.

Do not retrain the network or recompute PCA during inference.

## 9. Important output files

- `Model/trained_cnn.mat` — final baseline network, using the filename required by the assignment.
- `Results/preprocessing_config.mat` and/or `Results/prepared_datastores.mat` — saved preprocessing settings; include the file actually loaded by the prediction script.
- `Results/pca_band_selection.mat` — PCA results and selected bands.
- `Results/all_models_test_summary.csv` — overall test metrics.
- `Results/all_models_test_class_metrics.csv` — class-wise test metrics.
- `Results/final_experiment_metrics.csv` — consolidated metrics including mean recall and macro precision.
- `Results/Plots/` — figures for the report.

