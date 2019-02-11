# Machine_Learning_on_Prostate_Cancer
Applying different machine learning algorithms on PCGA Prostate Cancer Gene Dataset for Feature Selection, Dimensional Reduction and Classification and Regression

In this project, PCGA Prostate Cancer Gene Expression dataset and PCGA Clinical dataset is used to apply different machine learning techniques for selecting biomarkers or genes out of 60K genes that are relevant to predcit patients gleason score, T-Stage and Tumor Recurrence. Different machine learning algortihms have been implemented for feature selection, dimensionality reduction and target prediction. The overall model accuracy is around 96% for each of the three targets.

Merge_dataset.py: This script is written to merge required columns from prad_tcga_genes.xls and prad_tcga_clinical_data.xls dataset. As example, for gleason_score prediction, this sript will generate a new dataset where genes columns from prad_tcga_genes.xls file and gleason_score column from prad_tcga_clinical_data.xls file have been stacked with respect to each patient. Same way Predict_t_stage.csv and Tumor_Recurrence.csv datset can be made. More detail is available in the script.

Gleason_score.py: Predicts gleason_score based on genes using PCA, LDA and Random Forest Classifier. Model evaluation is done with 10-fold cross-validation.  Result is plotted using matplotlib. Confusion Matrix is generated to check number of TP, TN, FP and FN for each of the five classes. More detail is available in the script.

Predict_t_stage.py: Predicts t_stage based on genes using PCA, LDA and Random Forest Classifier. Model evaluation is done with 10-fold cross-validation.  Result is plotted using matplotlib. Confusion Matrix is generated to check number of TP, TN, FP and FN for the stages. More detail is available in the script.

TumorRecurrence.py: Predicts if a tumor will come back or not (0 or 1) based on genes using PCA, LDA and Random Forest Classifier. Model evaluation is done with 10-fold cross-validation.  Result is plotted using matplotlib. Confusion Matrix is generated to check number of TP, TN, FP and FN for the positive and negative classes. More detail is available in the script.

FS_with_random_forest.py: Uses Random Forest classifier for selecting top k features based on feature importance ranking. More detail is available in the script.

Information_Gain.py:  Uses Information Gain algorithm for selecting top k features based on feature IG score. More detail is available in the script.

LowVariance.py: Checks for features or columns whose values does not change significantly over the samples in the input dataset. These low variance features contribute less in predicting the target, thus can be safely discarded without hampering the accuracy.
