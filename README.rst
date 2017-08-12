What is logloss-beraf?
----------------------

A tool for selection of a limited number of informative DNA methylation
regions (i.e. sites) based on a combination of several feature selection
methods and an ensemble-based classifier. It is expected to handle higly
unbalanced and heterogeneous data. Also it is intended for the design
of diagnostic panels that can be potentially used in routine laboratory practice.

Quick start
-----------

1. `Install`_ ``logloss-beraf`` with all the dependencies::
    ```bash
    pip install logloss_beraf
    ```

2. `Make a test run`. It uses test data included to the package
    ```bash
    logloss_beraf test_run
    ```

3. `Prepare input feature and annotation tables.` The order of samples in those tables is supposed to be the same
    Methylation data
    ```
               Feature_1 Feature_2  Feature_3
    Sample_0   0.909642  0.823715   0.069785
    Sample_1   0.564799  0.199724   0.840741
    Sample_2   0.685081  0.489773   0.286591
    Sample_3   0.810637  0.006836   0.888038
    Sample_4   0.124098  0.347752   0.954853
    ```
    Annotation data
    ```
         Sample_Name  Type
    0    Sample_0     Benign
    1    Sample_1     Pathologic
    2    Sample_2     Benign
    3    Sample_3     Benign
    4    Sample_4     Pathologic
    ```

4. `Train model`
    ```bash
    logloss_beraf train \
      --features <path_to_feature_table> \
      --features_max_num 10 \
      --min_beta_threshold 0.2 \
      --annotation <path_to_annotation_table> \
      --sample_name_column "Sample_Name" \
      --class_column "Type" \
      --output_folder <path_to_output_folder>
    ```

5. `Apply trained model to independent dataset`
    ```bash
    logloss_beraf apply \
      --features <path_to_test_feature_table> \
      --model <path_to_trained_model>
      --output_folder <path_to_output_folder>
    ```


