# coding=utf-8
import copy
import logging
import os

# https://github.com/matplotlib/matplotlib/issues/3466/#issuecomment-195899517
import itertools
import matplotlib

matplotlib.use('agg')
import numpy as np
import pandas
from sklearn import (
    preprocessing,
    model_selection,
)
from sklearn.cross_validation import (
    LeaveOneOut,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RandomizedLogisticRegression
import cPickle as pickle

from utils.constants import (
    PREFILTER_PCA_PLOT_NAME,
    POSTFILTER_PCA_PLOT_NAME,
    FEATURE_IMPORTANCE_PLOT_NAME,
    FEATURE_COLUMN,
    FEATURE_IMPORTANCE_COLUMN,
    TRAINED_MODEL_NAME,
)
from visualization.plotting import plot_pca_by_annotation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from settings import logger


class LLBModelTrainer(object):
    """
    Class implementing main steps of the algorithm:
        1. Initial regions filtering with a user-specified delta beta-values threshold
        2. Applying randomized logistic regression in order to additionally pre-filter input regions
        3. Extracting highly correlated sites
        4. Reconstructing logloss function on the interval of user specified limit of number of sites
        5. Detecting optimal panel of regions and training final model
    Also does some visualizations
    """

    def __init__(self, threads=0, max_num_of_features=20,
                 cv_method="SKFold", class_weights="balanced", final_clf_estimators_num=3000,
                 intermediate_clf_estimators_num=1000, logloss_estimates=50, min_beta_threshold=0.2,
                 rr_iterations=5000, correlation_threshold=0.85, output_folder=None):
        """
        :param threads:
        :type threads: int
        :param max_num_of_features: maximum number of features a model can contain
        :type max_num_of_features: int
        :param cv_method: Supported cross-validation methods: "LOO", "SKFold"
        :type cv_method: str
        :param class_weights: Class balancing strategy
        :type class_weights: dict, str
        :param final_clf_estimators_num: number of estimators used in a final classifier
        :type final_clf_estimators_num: int
        :param intermediate_clf_estimators_num: number of estimators used in intermediate classifiers
        :type intermediate_clf_estimators_num: int
        :param logloss_estimates:  Number of LogLoss estimates on number of sites limited interval
        :type logloss_estimates: int
        :param min_beta_threshold: Minimum beta-values difference threshold
        :type min_beta_threshold: float
        :param rr_iterations: Number of randomized regression iterations
        """
        self.threads = threads
        self.max_num_of_features = max_num_of_features
        self.min_beta_threshold = min_beta_threshold

        # train process configuration
        self.cv_method = cv_method
        self.class_weights = class_weights
        self.final_clf_estimators_num = final_clf_estimators_num
        self.intermediate_clf_estimators_num = intermediate_clf_estimators_num
        self.rr_iterations = rr_iterations

        self.logloss_estimates = logloss_estimates

        # common
        self.correlation_threshold = correlation_threshold
        self.output_folder = output_folder if output_folder is not None else "results"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _run_randomized_regression(self, feature_df, annotation, clinical_column, sample_fraction=0.7):
        annotation = copy.deepcopy(annotation)
        # Encode labels of the classes
        le = preprocessing.LabelEncoder()
        annotation[clinical_column] = le.fit_transform(annotation[clinical_column])

        clf = RandomizedLogisticRegression(
            n_resampling=self.rr_iterations,
            sample_fraction=sample_fraction,
            n_jobs=1,
            verbose=1,
        ).fit(feature_df, annotation[clinical_column])

        selected_features = feature_df.T[clf.scores_ != 0].index
        logger.info("Number of selected features: %d", len(selected_features))
        return selected_features, clf

    def _train_clf(self, X, y, n_estimators=10):
        clf = RandomForestClassifier(n_estimators, n_jobs=self.threads, class_weight=self.class_weights)
        scores = scores_accuracy = np.array([0])

        cv_algo = None
        if self.cv_method is not None:
            if self.cv_method == "LOO":
                cv_algo = LeaveOneOut(len(y))
            elif self.cv_method == "SKFold":
                cv_algo = StratifiedKFold(y)

            logger.info("Running cross-validation...")
            scores = model_selection.cross_val_score(
                clf,
                X,
                y,
                cv=cv_algo,
                scoring='neg_log_loss',
                n_jobs=self.threads,
                verbose=1,
            )

        clf.fit(X, y)
        return clf, scores.mean(), scores.std()

    def _describe_and_filter_regions(self, basic_region_df, annotation, clinical_column, sample_name_column):
        logger.info("Initial number of regions: {0}".format(basic_region_df.shape))

        # Initial filtering based on min_beta_threshold
        class_combinations = itertools.combinations(annotation[clinical_column].unique(), 2)
        for combination in class_combinations:
            first_class_samples = annotation[annotation[clinical_column] == combination[0]][sample_name_column]
            second_class_samples = annotation[annotation[clinical_column] == combination[1]][sample_name_column]
            mean_difference = (basic_region_df.loc[first_class_samples].mean()
                               - basic_region_df.loc[second_class_samples].mean())
            basic_region_df = basic_region_df[mean_difference[abs(mean_difference) > self.min_beta_threshold].index.tolist()]
            basic_region_df = basic_region_df.dropna(how="any", axis=1)

        logger.info("Number of features after initial filtration: {0}".format(basic_region_df.shape))

        plot_pca_by_annotation(
            basic_region_df,
            annotation,
            clinical_column,
            sample_name_column,
            outfile=os.path.join(self.output_folder, PREFILTER_PCA_PLOT_NAME),
        )

        logger.info("Starting feature selection with RLR...")
        selected_features, model = self._run_randomized_regression(
            basic_region_df,
            annotation,
            clinical_column,
        )
        plot_pca_by_annotation(
            basic_region_df[selected_features],
            annotation,
            clinical_column,
            sample_name_column,
            outfile=os.path.join(self.output_folder, POSTFILTER_PCA_PLOT_NAME),
        )

        return selected_features, model

    def plot_fi_distribution(self, feature_importances):
        ax = feature_importances[FEATURE_IMPORTANCE_COLUMN].hist()
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Number of features")
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.output_folder, FEATURE_IMPORTANCE_PLOT_NAME))

    def _apply_feature_imp_thresh(self, features, feature_imp, thresh):
        return [
            feature[0] for feature in
            zip(features.values, feature_imp)
            if feature[1] > thresh
        ]

    def get_threshold(self, logloss_df):
        # Standard error
        ll_se = logloss_df["mean"].std() / np.sqrt(len(logloss_df["mean"]))

        # Restricting search to desired number of features.
        logloss_df = logloss_df[logloss_df["len"] <= int(self.max_num_of_features)]

        ll_max = logloss_df[logloss_df["mean"] == logloss_df["mean"].max()].iloc[0]
        ll_interval = logloss_df[logloss_df["mean"] > (ll_max["mean"] - 0.5 * ll_se)]
        res = ll_interval[ll_interval["len"] == ll_interval["len"].min()].iloc[0]
        return res

    def train(self, train_regions, anndf, sample_class_column, sample_name_column):
        """
        Main functionality
        :param train_regions: input dataframe with all regions methylation
        :type train_regions: pandas.DataFrame
        :param anndf: annotation dataframe, containing at least sample name and sample class
        :type anndf: pandas.DataFrame
        :param sample_class_column: name of the sample class column
        :type sample_class_column: str
        :param sample_name_column: name of the sample name column
        :type sample_name_column: str
        :return:
        """
        # train_regions = train_regions.T

        # First sort both train_regions and annotation according to sample names
        train_regions = train_regions.sort_index(ascending=True)
        # Ensure annotation contains only samples from the train_regions
        anndf = anndf[anndf[sample_name_column].isin(train_regions.index.tolist())].sort_values(
            by=[sample_name_column],
            ascending=True
        ).dropna(subset=[sample_name_column])
        train_regions = train_regions.ix[anndf[sample_name_column].tolist()]
        assert anndf[sample_name_column].tolist() == train_regions.index.tolist(), \
            "Samples in the annotations table are diferrent from those in feature table"

        # Prefilter regions
        selected_regions, clf = self._describe_and_filter_regions(
            train_regions,
            anndf,
            sample_class_column,
            sample_name_column,
        )

        # Estimate feature importances (FI)
        first_clf, mean, std = self._train_clf(
            train_regions[selected_regions.values],
            anndf[sample_class_column],
            n_estimators=self.final_clf_estimators_num,
        )

        feature_importances = pandas.DataFrame.from_records(
            zip(selected_regions.values, first_clf.feature_importances_),
            columns=[FEATURE_COLUMN, FEATURE_IMPORTANCE_COLUMN],
        )

        # Visualizing feature importance distribution
        self.plot_fi_distribution(feature_importances)

        # Extracting correlated site
        feature_importances = feature_importances[
            abs(feature_importances[FEATURE_IMPORTANCE_COLUMN]) > 0
            ]

        corr_matrix = train_regions[feature_importances[FEATURE_COLUMN]].corr().applymap(
            lambda x: 1 if abs(x) >= self.correlation_threshold else 0
        )

        logloss_df_cols = ["thresh", "mean", "std", "len"]
        logloss_di = pandas.DataFrame(columns=logloss_df_cols)

        for thresh in np.arange(
                feature_importances[FEATURE_IMPORTANCE_COLUMN].quantile(0.99),
                feature_importances[FEATURE_IMPORTANCE_COLUMN].max(),
                (
                        feature_importances[FEATURE_IMPORTANCE_COLUMN].max() -
                        feature_importances[FEATURE_IMPORTANCE_COLUMN].min()
                ) / self.logloss_estimates
        ):
            selected_features = self._apply_feature_imp_thresh(selected_regions, first_clf.feature_importances_, thresh)
            if len(selected_features) < 2:
                continue

            logger.info(
                "Estimating %d features on feature importance threshold %f",
                len(selected_features),
                thresh
            )
            clf, mean, std = self._train_clf(
                train_regions[selected_features],
                anndf[sample_class_column],
                n_estimators=self.intermediate_clf_estimators_num,
            )
            logloss_di = logloss_di.append(
                pandas.Series([thresh, mean, std, len(selected_features)], index=logloss_df_cols),
                ignore_index=True,
            )
            logger.info("LogLoss mean=%f, std=%f on threshold %f", mean, std, thresh)

        logger.info("Detecting optimal feature subset...")
        thresh = self.get_threshold(logloss_di)
        logger.info("Selected threshold")
        logger.info(thresh)
        selected_features = self._apply_feature_imp_thresh(
            selected_regions,
            first_clf.feature_importances_,
            thresh["thresh"],
        )

        logger.info("Trainig final model...")
        clf, mean, std = self._train_clf(
            train_regions[selected_features],
            anndf[sample_class_column],
            n_estimators=self.final_clf_estimators_num,
        )
        logger.info("Selected features: {0}".format(selected_features))

        pickle.dump((clf, selected_features), open(os.path.join(self.output_folder, TRAINED_MODEL_NAME), 'w'))

        return selected_features, clf, mean, std
