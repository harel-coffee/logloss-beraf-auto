import pandas
import pytest
from sklearn.datasets import make_classification

# Build a classification task using 3 informative features
from logloss_beraf.model_ops.trainer import LLBModelTrainer

SAMPLE_NAME_COLUMN = "Sample_Name"
CLINICAL_COLUMN = "Type"

X, y = make_classification(
    n_samples=200,
    n_features=100,
    n_informative=3,
    n_redundant=60,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)

annotation = pandas.DataFrame({
    SAMPLE_NAME_COLUMN: ["Sample_" + str(i) for i in range(X.shape[0])],
    CLINICAL_COLUMN: ["First class" if i == 0 else "Second class" for i in y],
})

OUTPUT_FOLDER = "./out"

sites = pandas.DataFrame(X, columns=["Feature_" + str(i) for i in range(0, 100, 1)])
sites.index = sites.index.map(lambda x: "Sample_" + str(x))

def test_informative_feature_selection():
    trainer = LLBModelTrainer(
        max_num_of_features=5,
        intermediate_clf_estimators_num=10,
        final_clf_estimators_num=10,
        rr_iterations=20,
        threads=1,
        output_folder=OUTPUT_FOLDER,
    )
    selected_features, clf, mean, std = trainer.train(
        sites,
        annotation,
        sample_class_column=CLINICAL_COLUMN,
        sample_name_column=SAMPLE_NAME_COLUMN,
    )
    assert len(selected_features) == 3
