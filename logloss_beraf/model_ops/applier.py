import cPickle as pickle

import logging

logger = logging.getLogger(__name__)

class LLBModelApplier(object):

    def __init__(self, output_folder=None):
        self.output_folder = output_folder

    def apply(self, features, model_path):
        classifier, selected_features = pickle.load(open(model_path, 'r'))
        logger.info("Predicted classes:")
        logger.info(classifier.predict(features))
