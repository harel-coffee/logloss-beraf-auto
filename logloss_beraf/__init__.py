import argparse
import os
import yaml
import pandas


def _check_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError("Path %s does not exist" % path)
    return path

def _parse_config(path, block):
    data = yaml.load(open(path, 'r'))
    return data[block]

def train(args):
    # for faster arguments validations
    from logloss_beraf.model_ops.trainer import LLBModelTrainer

    config = _parse_config(args.config, "train")
    config["threads"] = args.threads
    config["output_folder"] = args.output_folder
    config["max_num_of_features"] = args.features_max_num

    features = pandas.read_csv(args.features, index_col=0)
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
    if args.test is not None and (args.sample_name_column is None or args.class_column):
        raise argparse.ArgumentTypeError("If --test argument is specified, "
                                         "one should also provide --sample_name_colum an --class_column")
    # for faster arguments validations
    from model_ops.applier import LLBModelApplier
    config = {
        "output_folder": args.output_folder
    }
    features = pandas.read_csv(args.features, index_col=0)
    applier = LLBModelApplier(**config)
    applier.apply(features, args.model)


def test_run(args):
    command = ('logloss_beraf train '
               '--features \"{0}/resources/test_features.csv\" '
               '--features_max_num 5 '
               '--min_beta_threshold 0.2 '
               '--annotation \"{0}/resources/test_annotation.csv\" '
               '--sample_name_column Sample_Name '
               '--class_column Type '
               '--config \"{0}/resources/test_config.yaml\"'.format(os.path.dirname(__file__)))

    os.system(command)

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
        help="Maximum number of features a model can use",
    )
    train_parser.add_argument(
        "--min_beta_threshold",
        required=False,
        default=0.2,
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

    test_parser = subparsers.add_parser("test_run")
    test_parser.set_defaults(func=test_run)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
