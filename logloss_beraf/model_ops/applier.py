import cPickle as pickle
import logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn import metrics, preprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LLBModelApplier(object):

    def __init__(self, output_folder=None):
        self.output_folder = output_folder

    def apply(self, features, model_path, answers=None):
        """

        :param features:
        :param model_path:
        :param answers: if specified, test prediction on provided answers, plotting AUC and providing
        classification metrics
        :return:
        """
        report = None
        classifier, selected_features = pickle.load(open(model_path, 'r'))

        # Predicting classes for all input samples
        prediction = classifier.predict(features[selected_features])
        logger.info("Predicted classes:\n{0}".format(zip(features.index, prediction)))

        clf_proba = classifier.predict_proba(features[selected_features])
        logger.info("Prediction propabilities:\n{0}".format(clf_proba))

        if answers is not None:
            # Encode labels of the classes
            #le = preprocessing.LabelEncoder()
            #answers = le.fit_transform(answers)

            # ensure all samples have answers
            answers = answers.dropna()
            features = features.ix[list(set(answers.index).intersection(features.index))]
            answers = answers.ix[features.index]
            prediction = classifier.predict(features[selected_features])

            report = classification_report(answers, prediction)
            logger.info(report)

            auc = metrics.roc_auc_score(answers, clf_proba[:, 1])
            logger.info("AUC = {0}".format(auc))
        return report
