import argparse
import logging
import os
import random

import pandas
import yaml

from utils.feature_selection.stability import estimate_total_kuncheva_index

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
from settings import logger
import cPickle as pickle


def _check_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Path %s does not exist" % path)
    return path


def _parse_config(path, block):
    data = yaml.load(open(path, 'r'))
    return data[block]


def _read_csv(path, usecols=None, low_memory=False):
    return pandas.read_csv(
        path,
        index_col=0 if usecols is None else False,
        compression="gzip" if path.endswith(".gz") else None,
        low_memory=low_memory,
        na_filter=False,
        usecols=usecols
    )


def train(args, features=None):
    # for faster arguments validations
    from model_ops.trainer import LLBModelTrainer

    config = _parse_config(args.config, "train")
    config["threads"] = args.threads
    config["output_folder"] = args.output_folder
    config["max_num_of_features"] = args.features_max_num
    config["min_beta_threshold"] = args.min_beta_threshold

    if features is None:
        features = _read_csv(args.features)

    annotation = pandas.read_csv(args.annotation)

    trainer = LLBModelTrainer(**config)
    selected_features, clf, mean, std = trainer.train(
        features,
        annotation,
        sample_class_column=args.class_column,
        sample_name_column=args.sample_name_column,
    )
    return selected_features, clf, mean, std


def apply_model(args):
    if args.test is not None and (args.sample_name_column is None or args.class_column is None):
        raise argparse.ArgumentTypeError("If --test argument is specified, "
                                         "one should also provide --sample_name_colum an --class_column")
    answers = None
    # for faster arguments validations
    from model_ops.applier import LLBModelApplier
    config = {
        "output_folder": args.output_folder
    }

    classifier, selected_features = pickle.load(open(args.model, 'r'))
    features = _read_csv(args.features, low_memory=False) #, usecols=selected_features)
    if args.test is not None:
        answers = pandas.read_csv(args.test, index_col=0).set_index(args.sample_name_column)[args.class_column]
        answers = answers.ix[features.index]
    applier = LLBModelApplier(**config)
    return applier.apply(features, args.model, answers=answers)


def estimate_stability(args):
    """
    Estimates the stability of the feature selection algorithm and forms a report
    :param args: the same as input for train but with additional number of bootstrapping iterations parameter
    :return:
    """
    features = _read_csv(args.features)
    feature_subsets = []
    for i in range(args.bootstrap_iterations):
        cur_features = features.ix[random.sample(features.index, int(args.sampling * features.shape[0]))]
        feature_subsets.append(set(train(args, features=cur_features)[0]))

    index = estimate_total_kuncheva_index(feature_subsets, 1000)
    logger.info("Stability index for {0} runs is {1}".format(args.bootstrap_iterations, index))
    return index


def test_run(args):
    command = ('python ./logloss_beraf/__init__.py stability '
               '--features \"{0}/resources/test_features.csv\" '
               '--features_max_num 5 '
               '--min_beta_threshold 0.2 '
               '--annotation \"{0}/resources/test_annotation.csv\" '
               '--sample_name_column Sample_Name '
               '--class_column Type '
               '--bootstrap_iterations 10 --sampling 0.9'.format(os.path.dirname(__file__)))

    logger.info("Initiating command \n %s" % command)

    os.system(command)

    logger.info("Test run finished successfuly")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument(
        "--features",
        type=_check_path,
        required=True,
        help=("Path to the table with per sample per region feature values. Rows are treated as samples,"
              "columns as features. The order of samples should be the same as in the annotation file")
    )
    train_parser.add_argument(
        "--features_max_num",
        required=True,
        type=int,
        help="Maximum number of features a model can use",
    )
    train_parser.add_argument(
        "--min_beta_threshold",
        required=False,
        default=0.2,
        type=float,
        help="Minimum difference between mean values of feature in classes",
    )
    train_parser.add_argument(
        "--annotation",
        type=_check_path,
        required=True,
        help="Path to the sample annotation table",
    )
    train_parser.add_argument(
        "--sample_name_column",
        required=True,
        help="Name of the sample name column in the annotation table",
    )
    train_parser.add_argument(
        "--class_column",
        required=True,
        help="Name of the class column in the annotation table",
    )
    train_parser.add_argument(
        "--config",
        type=_check_path,
        required=False,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Configuration file with \"low level\" training options"
    )
    train_parser.add_argument(
        "--output_folder",
        required=False,
        default="output",
        help="Path to the folder with results",
    )
    train_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=1,
        help="Number of threads",
    )

    apply_parser = subparsers.add_parser("apply")
    apply_parser.set_defaults(func=apply_model)
    apply_parser.add_argument(
        "--features",
        type=_check_path,
        required=True,
        help=("Path to the table with per sample per region feature values. Rows are treated as samples,"
              "columns as features. The order of samples should be the same as in the annotation file")
    )
    apply_parser.add_argument(
        "--model",
        required=True,
        type=_check_path,
        help="Path to the trained model",
    )
    apply_parser.add_argument(
        "--test",
        required=False,
        type=_check_path,
        help=("Path to the sample annotation file. If specified, tests model according to annotation file,"
              "outputs classification metrics and plots AUC"),
    )
    apply_parser.add_argument(
        "--sample_name_column",
        required=False,
        help="Name of the sample name column in the annotation table",
    )
    apply_parser.add_argument(
        "--class_column",
        required=False,
        help="Name of the class column in the annotation table",
    )

    apply_parser.add_argument(
        "--config",
        type=_check_path,
        required=False,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Configuration file with \"low level\" training options"
    )
    apply_parser.add_argument(
        "--output_folder",
        required=False,
        default="output",
        help="Path to the folder with results",
    )

    stability_parser =subparsers.add_parser("stability")
    stability_parser.set_defaults(func=estimate_stability)
    stability_parser.add_argument(
        "--features",
        type=_check_path,
        required=True,
        help=("Path to the table with per sample per region feature values. Rows are treated as samples,"
              "columns as features. The order of samples should be the same as in the annotation file")
    )
    stability_parser.add_argument(
        "--features_max_num",
        required=True,
        type=int,
        help="Maximum number of features a model can use",
    )
    stability_parser.add_argument(
        "--min_beta_threshold",
        required=False,
        default=0.2,
        type=float,
        help="Minimum difference between mean values of feature in classes",
    )
    stability_parser.add_argument(
        "--annotation",
        type=_check_path,
        required=True,
        help="Path to the sample annotation table",
    )
    stability_parser.add_argument(
        "--sample_name_column",
        required=True,
        help="Name of the sample name column in the annotation table",
    )
    stability_parser.add_argument(
        "--class_column",
        required=True,
        help="Name of the class column in the annotation table",
    )
    stability_parser.add_argument(
        "--config",
        type=_check_path,
        required=False,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Configuration file with \"low level\" training options"
    )
    stability_parser.add_argument(
        "--output_folder",
        required=False,
        default="output",
        help="Path to the folder with results",
    )
    stability_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=1,
        help="Number of threads",
    )
    stability_parser.add_argument(
        "--bootstrap_iterations",
        required=True,
        type=int,
        default=50,
        help="Number of bootstrap iterations",
    )
    stability_parser.add_argument(
        "--sampling",
        required=True,
        type=float,
        default=0.9,
        help="Sampling rate",
    )

    test_parser = subparsers.add_parser("test_run")
    test_parser.set_defaults(func=test_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    main()
