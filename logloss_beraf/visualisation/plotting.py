# coding=utf-8
from __future__ import division

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_file, output_notebook
from bokeh.plotting import (
    show,
    save,
    figure,
)
import pandas
from math import sqrt
from scipy.stats import chisquare
from scipy import stats
import statsmodels
from sklearn.decomposition import PCA
from bokeh.models import HoverTool, ColumnDataSource
from collections import OrderedDict
import copy
import itertools
import seaborn

import logging
logger = logging.getLogger(__name__)


def plot_pca_two_group(group1, group2, title, group1_annotation, group2_annotation, outfile=None):
    logger.info("Preparing PCA plot...")
    t_group = group1.join(group2)
    pca = PCA(n_components=2)
    X_r = pca.fit(t_group.T.values).transform(t_group.T.values)

    logger.info("Explained variance ratio: {0}".format(pca.explained_variance_ratio_))

    p = figure(title=title, background_fill="#E8DDCB", tools="save,hover,resize")
    p.xaxis.axis_label = 'Component 1'
    p.yaxis.axis_label = 'Component 2'

    X_c = pandas.Series(X_r[:, 0])
    Y_c = pandas.Series(X_r[:, 1])

    group1_pca_source = ColumnDataSource(
        data=dict(
            x=X_c.values[0:group1.shape[1]],
            y=Y_c.values[0:group1.shape[1]],
            pid=group1.columns
        )
    )

    group2_pca_source = ColumnDataSource(
        data=dict(
            x=X_c.values[group1.shape[1]:group1.shape[1] + group2.shape[1]],
            y=Y_c.values[group1.shape[1]:group1.shape[1] + group2.shape[1]],
            pid=group2.columns
        )
    )

    p.scatter("x", "y", source=group1_pca_source,
              color="red", fill_alpha=0.2, size=10, legend=group1_annotation)
    p.scatter("x", "y", source=group2_pca_source,
              color="green", fill_alpha=0.2, size=10, legend=group2_annotation)

    p.legend.orientation = "vertical"
    hover = p.select(dict(type=HoverTool))

    hover.tooltips = OrderedDict([
        ("Participant ID", "@pid"),
    ])
    if outfile is not None:
        output_file(outfile, mode='inline')
        save(p)



def plot_pca_by_annotation(df, annotation, clinical_column, sample_name_column, outfile=None):
    tdf_columns = df.columns
    tdf = df.merge(annotation, right_on=sample_name_column, left_index=True).set_index(sample_name_column, drop=True)
    groups = tdf[clinical_column].unique()
    for group1, group2 in itertools.combinations(groups, 2):
        clin_group1 = tdf[tdf[clinical_column] == group1]
        clin_group2 = tdf[tdf[clinical_column] == group2]
        plot_pca_two_group(
            clin_group1[tdf_columns].transpose(),
            clin_group2[tdf_columns].transpose(),
            clinical_column,
            str(group1),
            str(group2),
            outfile=outfile,
        )


def plot_pca_several_group(t_group, title, annotation, clinical_column=None, sample_column=None):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from bokeh.palettes import Spectral6, Accent8
    from operator import itemgetter

    output_notebook()
    # output_file(pipeline + "/PCA.TN.html")
    t_group = t_group.reset_index(drop=True)
    pca = PCA(n_components=5)
    X_r = pca.fit(t_group.T.values).transform(t_group.T.values)
    logger.info("Explained variance ratio:", pca.explained_variance_ratio_)
    p = figure(title=title, background_fill="#E8DDCB", tools="save,hover,resize")
    p.xaxis.axis_label = 'Component 1'
    p.yaxis.axis_label = 'Component 2'

    X_c = pandas.Series(X_r[:, 0])
    Y_c = pandas.Series(X_r[:, 1])
    margin = 0
    colors = Accent8

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(pandas.DataFrame(X_c), annotation[clinical_column])

    color_index = 0
    clinical_type_idx = 0
    if clinical_column is not None:
        for clinical_type in annotation[clinical_column].unique().tolist():
            current_group = t_group[annotation[annotation[clinical_column] == clinical_type][sample_column]]
            current_group_samples_index = [i for i, val in enumerate(t_group.columns.tolist()) if
                                           val in current_group.columns]

            logger.debug("Current group: ", clinical_type)
            logger.debug(current_group)
            logger.debug(current_group_samples_index)

            group_pca_source = ColumnDataSource(
                data=dict(
                    x=X_c.loc[current_group_samples_index].values,
                    y=Y_c.loc[current_group_samples_index].values,
                    pid=current_group.columns
                )
            )

            p.scatter(
                "x",
                "y",
                source=group_pca_source,
                fill_alpha=0.2,
                size=10,
                legend=clinical_type,
                color=colors[color_index],
            )

            min_x = np.min(X_c.loc[current_group_samples_index].values)
            max_x = np.max(X_c.loc[current_group_samples_index].values)

            min_y = np.min(Y_c.loc[current_group_samples_index].values)
            max_y = np.max(Y_c.loc[current_group_samples_index].values)

            clinical_type_idx += 1

            if color_index < len(colors) - 1:
                color_index += 1
            else:
                color_index = 0
    else:
        t_group_pca_source = ColumnDataSource(
            data=dict(
                x=X_c,
                y=Y_c,
                pid=t_group.columns
            )
        )
        p.scatter("x", "y", source=t_group_pca_source,
                  color="red", fill_alpha=0.2, size=10)

    p.legend.orientation = "vertical"
    hover = p.select(dict(type=HoverTool))

    hover.tooltips = OrderedDict([
        ("Participant ID", "@pid"),
    ])

    show(p)
