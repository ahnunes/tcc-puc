#!/usr/bin/env python

"""
Calibrate machine learn models using user reviews from products and services retrieved from websites.

Steps:

1 - Load data, filter and normalize it
2 - Run classifiers with default configuration
3 - Select bext classifiers and optimize hyperparams
4 - Combine classifiers to improce accurancy
"""

import hashlib
import heapq
import itertools
import logging
import multiprocessing
import os
import sys
import warnings
from time import time

import numpy as np
import pandas as pd
import spacy

import joblib
from joblib import Parallel, delayed, parallel_backend

import nltk
from nltk.stem import RSLPStemmer

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, RobustScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from whatthelang import WhatTheLang

from leia import SentimentIntensityAnalyzer

from aux import (load_spell_checker, df_parallel_apply, leia_sentiment, normalize_text, spell_check, remove_words,
                 heatmap_plot, most_informative_features, group_df, combined_score, flatten_list, sentiment, language,
                 stemmed_text, data_plots, get_stop_words, time_format, classifier_plots, wordcloud_plot, cached_instance,
                 REMOVED_STOP_WORDS, CPU_COUNT)


def load_df(stop_words, spell):
    """Return pandas data frame with normalized and filtered reviews"""

    filename = "data/df_filtered.jbl"

    if os.path.exists(filename):
        df = joblib.load(filename)
        return df

    wtl = WhatTheLang()
    stemmer = RSLPStemmer()

    # Raw data
    logging.info("Loading data from CSV file")
    df = pd.read_csv("data/data.csv").drop_duplicates()
    logging.info("Total rows: %d", len(df))

    df["text"] = df.apply(lambda row: f"{row.title} {row.text}", axis=1)
    del df["title"]

    logging.info("Running LEIA")
    s = SentimentIntensityAnalyzer()
    df["leia"] = df_parallel_apply(df, lambda row: leia_sentiment(row.text, analyzer=s))

    # Normalize text and remove stop words
    logging.info("Normalize text and remove stop words")
    df["text"] = df_parallel_apply(df, lambda row: normalize_text(row.text))
    df["text"] = df_parallel_apply(df, lambda row: spell_check(row.text, spell), series=128)
    df["text"] = df_parallel_apply(df, lambda row: remove_words(row.text, stop_words))

    common_stop_words = {k: v for k, v in sorted(REMOVED_STOP_WORDS.items(), key=lambda item: -item[1])}
    logging.info("Common stop words: %s", list(itertools.islice(common_stop_words, 50)))

    # Filter text with at least 10 characters and 5 words
    logging.info("Filter text with at least 10 characters and 5 words")
    df = df[(df["text"].str.len() > 10) & (df["text"].str.count(" ") >= 4)]
    logging.info("Total rows: %d", len(df))

    # Filter for portuguese text only
    logging.info("Filter for portuguese text only")
    df["language"] = df_parallel_apply(df, lambda row: language(row.text, wtl), backend="threading")

    logging.info(df[df["language"] == "en"].head())
    logging.info(df[df["language"] == "pt"].head())

    df = df[df["language"] == "pt"]
    logging.info("Total rows: %d", len(df))

    # Add sentiment and length columns
    df["sentiment"] = df_parallel_apply(df, lambda row: sentiment(row.rating, row.source))

    leia_accuracy = accuracy_score(df["leia"].values, df["sentiment"].values)
    logging.info("Leia accuracy: %.2f%%", leia_accuracy * 100)

    wordcloud_plot(df, "before")

    # Text Stemming
    logging.info("Stemming text")
    df["text"] = df_parallel_apply(df, lambda row: stemmed_text(row.text, stemmer=stemmer))

    # Filter text with at least 10 characters and 5 words
    logging.info("Filter text with at least 10 characters and 5 words")
    df = df[(df["text"].str.len() > 10) & (df["text"].str.count(" ") >= 4)]
    logging.info("Total rows: %d", len(df))

    wordcloud_plot(df, "after")

    df['length'] = df.apply(lambda row: len(row.text), axis=1)

    data_plots(df)

    joblib.dump(df, filename)
    logging.info("Saved '%s'", filename)

    return df


def best_voting_combination(voting_list, vectorizer, x_train_t, y_train_t, x_test_t, y_test):
    """Test all voting combinations and return the one with best score"""

    hashcode = hashlib.blake2s(
        str(([x[0] for x in voting_list], len(y_train_t), str(vectorizer))).encode("utf-8"),
        digest_size=8).hexdigest()

    filename = f"cache/VotingClassifier_{hashcode}.jbl"

    if os.path.exists(filename):
        return joblib.load(filename)

    size = len(voting_list)
    combinations = list(flatten_list(map(lambda x: list(itertools.combinations(range(size), x)), range(3, size + 1))))

    logging.info("VotingClassifier (%d)", len(combinations))

    t0 = time()

    with parallel_backend("loky"):
        scores = Parallel(n_jobs=CPU_COUNT, verbose=10)(
            delayed(combined_score)(combination, [voting_list[x][2] for x in combination], y_test) for combination in combinations
        )

    cutoff = min(heapq.nlargest(5, scores))
    indices = np.argwhere(scores >= cutoff).flatten().tolist()
    voting_time = time() - t0

    logging.info("Best accuracy: %.2f%% | Best indices: %s | Total time: %s",
                 max(scores) * 100, indices, time_format(voting_time, decimals=3))

    classifiers = [
        dict(estimators=[(str(y), voting_list[y][1]) for y in combinations[x]], n_jobs=CPU_COUNT, verbose=10)
        for x in indices]

    result = (-1,)

    for (i, values) in enumerate(classifiers):
        try:
            logging.info("VotingClassifier %d", i + 1)
            classifier = VotingClassifier(**values)

            t0 = time()

            classifier.fit(x_train_t, y_train_t)

            train_time = time() - t0
            logging.info("Train time: %s", time_format(train_time, decimals=3))

            t0 = time()

            predictions = classifier.predict(x_test_t)

            predict_time = time() - t0

            old_acc = scores[indices[i]]
            new_acc = accuracy_score(y_test, predictions)

            accuracy = (new_acc + 2 * old_acc) / 3  # Weighted average

            logging.info("Accuracy: %.2f%% / %.2f%%", accuracy * 100, old_acc * 100)

            if accuracy > result[0]:
                result = (accuracy, old_acc, classifier, combinations[indices[i]], train_time, predict_time)
        except Exception as e:  # pylint: disable=broad-except
            logging.error(e)

    if len(result) > 1:
        joblib.dump(result, filename)
        logging.info("Saved '%s'", filename)

    return result


def restrict_options(hashcode, vectorizers, classifiers, x_train, x_test, y_train, y_test):
    """Return only vectorizers and classifiers with scores greater than average"""

    filename = f"data/restrict_options_{len(y_train)}.jbl"

    if os.path.exists(filename):
        df_final = joblib.load(filename)
    else:
        table = []

        for (v_label, v_name, v_values) in vectorizers:
            vectorizer, _ = cached_instance(hashcode, v_name, v_values, x_train)

            x_train_t, y_train_t, x_test_t, _scaler = remove_outliers_and_normalize(
                hashcode,
                vectorizer,
                vectorizer.transform(x_train),
                y_train,
                vectorizer.transform(x_test)
            )

            for (family, name, values) in classifiers:
                c_name = f"{family}.{name}"
                print(f"{v_label} {name}")
                logging.info("%s %s", v_label, name)

                classifier, train_time = cached_instance(
                    hashcode + str(vectorizer), name, values, x_train_t, y_train_t, family=family
                )

                row = {
                    "v_name": v_label,
                    "c_name": c_name,
                    "accuracy": None,
                    "train_time": train_time,
                    "predict_time": -1,
                }

                try:
                    t0 = time()

                    predictions = classifier.predict(x_test_t)
                    accuracy = accuracy_score(y_test, predictions)

                    row["predict_time"] = time() - t0
                    row["accuracy"] = accuracy
                    logging.info("Accuracy: %.2f%%", 100 * accuracy)
                except Exception as e:  # pylint: disable=broad-except
                    logging.error(e)

                table.append(row)

                heatmap_plot(pd.DataFrame(table), filename="int")

        df_final = pd.DataFrame(table)
        joblib.dump(df_final, filename)
        logging.info("Saved '%s'", filename)

    return best_options(df_final, vectorizers, classifiers)


def best_options(df, vectorizers, classifiers):
    """Return vectorizers and classifiers with score greated than average"""

    df_grouped = group_df(df)

    avg_col = df_grouped["Average"]
    avg_row = df_grouped[df_grouped.index == "Average"]
    avg_value = avg_col["Average"]

    df_grouped = df_grouped[df_grouped["Average"] > avg_value]
    c_names = list(df_grouped.index)
    v_names = [x for x in list(df_grouped.columns) if avg_row[x].sum() > avg_value]

    c_names = [x.replace("catboost", "aux") for x in c_names]
    classifiers = [x for x in classifiers if f"{x[0]}.{x[1]}" in c_names]
    vectorizers = [x for x in vectorizers if x[0] in v_names]

    return vectorizers, classifiers


def remove_outliers_and_normalize(hashcode, vectorizer, x_train_t, y_train_t, x_test_t):
    """Remove outliers from train data and normalize both train and test data"""

    hashcode += str(vectorizer)

    if hasattr(x_train_t, "shape") and len(x_train_t.shape) > 1:
        hashcode += str((x_train_t.shape, x_train_t[:10].A[0]))
    else:
        hashcode += str((len(x_train_t), x_train_t[:10]))

    hashcode = hashlib.blake2s(hashcode.encode("utf-8"), digest_size=8).hexdigest()
    filename = f"cache/normalized_{hashcode}.jbl"

    if os.path.exists(filename):
        return joblib.load(filename)

    logging.info("Removing outliers and normalizing data")

    outlier_detection = LocalOutlierFactor(n_neighbors=5, contamination=.01)
    outliers = outlier_detection.fit_predict(x_train_t)

    x_train_t = x_train_t[outliers > 0]
    y_train_t = y_train_t[outliers > 0]

    scaler = RobustScaler(with_centering=False).fit(x_train_t)
    x_train_t = scaler.transform(x_train_t)
    x_test_t = scaler.transform(x_test_t)

    logging.info("Outliers: %d", len(outliers[outliers < 0]))

    # Mandatory for some classifiers, e.g. KerasClassifier
    x_train_t.sort_indices()
    x_test_t.sort_indices()

    result = (x_train_t, y_train_t, x_test_t, scaler)

    joblib.dump(result, filename)
    logging.info("Saved '%s'", filename)

    return result


def best_option(hashcode, vectorizers, classifiers, x_train, x_test, y_train, y_test):
    """Return best vectorizer, scaler and classifier"""
    filename = f"cache/best_{len(x_train)}.jbl"

    if os.path.exists(filename):
        return joblib.load(filename)

    logging.info("Train size: %d", len(x_train))

    vectorizers, classifiers = restrict_options(hashcode, vectorizers, classifiers, x_train, x_test, y_train, y_test)
    table = []

    best_classifier = (0,)

    for (v_label, v_name, v_values) in vectorizers:
        vectorizer, _ = cached_instance(hashcode, v_name, v_values, x_train)

        x_train_t, y_train_t, x_test_t, scaler = remove_outliers_and_normalize(
            hashcode,
            vectorizer,
            vectorizer.transform(x_train),
            y_train,
            vectorizer.transform(x_test)
        )

        info_gain = dict(zip(
            vectorizer.get_feature_names(),
            mutual_info_classif(x_train_t, y_train_t, discrete_features=True)
        ))

        info_gain = {k: v for k, v in sorted(info_gain.items(), key=lambda item: -item[1])[:20]}

        logging.info("%s Information Gain %s", v_label, info_gain)

        voting_list = []

        for (family, name, values) in classifiers:
            c_name = f"{family}.{name}"
            print(f"{v_label} {name} (GridSearchCV)")
            logging.info("%s %s (GridSearchCV)", v_label, name)

            values = {"cv": 2, "n_jobs": CPU_COUNT, **values}
            row = {
                "v_name": v_label,
                "c_name": c_name,
                "accuracy": None,
                "train_time": -1,
                "predict_time": -1,
                "flag": "",
            }

            try:
                classifier, row["train_time"] = cached_instance(
                    hashcode + str(vectorizer), name, values, x_train_t, y_train_t, family=family, grid_search=True
                )

                t0 = time()

                if hasattr(classifier, "best_params_"):
                    logging.info(classifier.best_params_)
                    classifier = classifier.best_estimator_

                predictions = classifier.predict(x_test_t)
                accuracy = accuracy_score(y_test, predictions)

                voting_list.append((c_name, classifier, predictions))

                row["predict_time"] = time() - t0
                row["accuracy"] = accuracy
                logging.info("Accuracy: %.2f%%", 100 * accuracy)
            except Exception as e:  # pylint: disable=broad-except
                logging.error(e)

            table.append(row)

            heatmap_plot(pd.DataFrame(table), filename="final", average=False)

        _cur_acc, accuracy, voting, combination, train_time, predict_time = best_voting_combination(
            voting_list, vectorizer, x_train_t, y_train_t, x_test_t, y_test
        )

        best_params = [voting_list[x][0] for x in map(lambda x: int(x[0]), voting.named_estimators_)]
        logging.info("%s %.2f%%", best_params, 100 * accuracy)

        table.append({
            "v_name": v_label,
            "c_name": "VotingClassifier",
            "accuracy": accuracy,
            "train_time": train_time,
            "predict_time": predict_time,
            "flag": "",
        })

        if accuracy > best_classifier[0]:
            best_classifier = (accuracy, voting, vectorizer, scaler)

        for i in combination:
            for row in table:
                if row['c_name'] == voting_list[i][0] and row['v_name'] == v_label:
                    row['flag'] = "*"
                    break

        heatmap_plot(pd.DataFrame(table), filename="final", average=False, time_heatmap=True)

    df_final = pd.DataFrame(table)
    joblib.dump(df_final, f"data/df_final_{len(x_train)}.jbl")
    logging.info("Saved 'data/df_final_%d.jbl'", len(x_train))

    joblib.dump(best_classifier, filename)
    logging.info("Saved '%s'", filename)

    return best_classifier


def keras_model(optimizer="Adamax", activation="softplus", units=32):
    """Function to create model, required for KerasClassifier"""
    model = Sequential()
    model.add(Dense(units, activation="relu", input_dim=2500))
    model.add(Dense(2, activation=activation))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def main():
    """Main entry point"""

    spell = load_spell_checker()
    stemmer = RSLPStemmer()
    stop_words = get_stop_words()

    df = load_df(stop_words, spell)

    texts = df["text"].values
    encoder = LabelEncoder()
    sentiments = encoder.fit_transform(df["sentiment"].values)
    sentiments_map = dict(zip(sentiments, df["sentiment"].values))
    labels = encoder.classes_
    train_size = 0.5
    hashcode = hashlib.blake2s(str(texts[:10]).encode("utf-8"), digest_size=8).hexdigest()

    vectorizers = [
        ("Unigram", "CountVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, binary=False)
         ),

        ("Bigram", "CountVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, binary=False, ngram_range=(1, 2))
         ),

        ("Trigram", "CountVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, binary=False, ngram_range=(1, 3))
         ),

        ("2-Skip-Bigram", "SkipGramVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, binary=False, k=2, ngram_range=(1, 2))
         ),

        ("2-Skip-Trigram", "SkipGramVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, binary=False, k=2, ngram_range=(1, 3))
         ),

        ("TF-IDF Unigram", "TfidfVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8)
         ),

        ("TF-IDF Bigram", "TfidfVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, ngram_range=(1, 2))
         ),

        ("TF-IDF Trigram", "TfidfVectorizer",
         dict(max_features=2500, min_df=8, max_df=0.8, ngram_range=(1, 3))
         ),
    ]

    classifiers = [
        ("aux", "KerasClassifier", dict(
            estimator=dict(
                build_fn=keras_model, epochs=5, batch_size=10
            ),
            param_grid=dict(
                epochs=[5, 10, 15],
                batch_size=[3, 5, 10],
            ),
        )),

        ("xgboost", "XGBClassifier", dict(
            estimator=dict(
                base_score=0.5, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                gamma=[1, 0.1, 0.001, 0.0001],
                max_depth=[5, 10],
            ),
        )),

        ("xgboost", "XGBRFClassifier", dict(
            estimator=dict(
                base_score=0.5, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                gamma=[1, 0.1, 0.001, 0.0001],
                max_depth=[5, 10],
            ),
        )),

        ("lightgbm", "LGBMClassifier", dict(
            estimator=dict(
                n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                learning_rate=[0.1, 0.5, 1.0],
                max_depth=[5, 6, 7, -1],
            ),
        )),

        ("aux", "CatBoostClassifier", dict(
            estimator=dict(
                n_estimators=100, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                learning_rate=[0.03, 0.5, 1.0],
                max_depth=[5, 6, 7],
            ),
        )),

        ("linear_model", "LogisticRegression", dict(
            estimator=dict(
                max_iter=2000, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                solver=["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                C=[0.05, 0.5, 1.0],
                penalty=["l1", "l2"],
            ),
        )),

        ("linear_model", "SGDClassifier", dict(
            estimator=dict(
                max_iter=2000, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                loss=["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                learning_rate=["optimal", "invscaling", "adaptive"],
                eta0=[0.01, 0.05, 0.1, 0.25, 0.75, 1],
            ),
        )),

        ("linear_model", "RidgeClassifier", dict(
            estimator=dict(
                max_iter=2000, random_state=0
            ),
            param_grid=dict(
                alpha=[0, 0.001, 0.01, 0.25, 0.5, 0.75, 1.0],
            ),
        )),

        ("linear_model", "PassiveAggressiveClassifier", dict(
            estimator=dict(
                max_iter=2000, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                loss=["hinge", "squared_hinge"],
            ),
        )),

        ("linear_model", "Perceptron", dict(
            estimator=dict(
                max_iter=2000, n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                penalty=["l2", "l1", "elasticnet", None],
                alpha=[0, 0.001, 0.01, 0.25, 0.5, 0.75, 1.0],
                eta0=[0.01, 0.05, 0.1, 0.25, 0.75, 1],
            ),
        )),

        ("tree", "DecisionTreeClassifier", dict(
            estimator=dict(
                random_state=0
            ),
            param_grid=dict(
                criterion=["gini", "entropy"],
                splitter=["best", "random"],
                min_samples_leaf=[1, 2, 5, 7],
                max_depth=[5, 6, 7, None],
            ),
        )),

        ("tree", "ExtraTreeClassifier", dict(
            estimator=dict(
                random_state=0
            ),
            param_grid=dict(
                criterion=["gini", "entropy"],
                splitter=["best", "random"],
                min_samples_leaf=[1, 2, 5, 7],
                max_depth=[5, 6, 7, None],
            ),
        )),

        ("naive_bayes", "MultinomialNB", dict(
            estimator=dict(),
            param_grid=dict(
                alpha=[0, 0.001, 0.01, 0.25, 0.5, 0.75, 1.0],
            ),
        )),

        ("naive_bayes", "BernoulliNB", dict(
            estimator=dict(),
            param_grid=dict(
                alpha=[0.25, 0.5, 0.75, 1.0],
                binarize=[0.0, 0.5, 1.0]
            ),
        )),

        ("ensemble", "RandomForestClassifier", dict(
            estimator=dict(
                n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                criterion=["entropy", "gini"],
                min_samples_leaf=[1, 2, 5, 7],
                max_depth=[5, 6, 7, None],
            ),
        )),

        ("ensemble", "ExtraTreesClassifier", dict(
            estimator=dict(
                n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[100, 200],
                criterion=["entropy", "gini"],
                min_samples_leaf=[1, 2, 5, 7],
                max_depth=[5, 6, 7, None],
            ),
        )),

        ("ensemble", "GradientBoostingClassifier", dict(
            estimator=dict(
                n_estimators=100, random_state=0
            ),
            param_grid=dict(
                loss=["deviance", "exponential"],
                criterion=["friedman_mse", "mse"],
                min_samples_leaf=[1, 3, 7],
                max_depth=[3, 5, 7],
            ),
        )),

        ("ensemble", "BaggingClassifier", dict(
            estimator=dict(
                n_jobs=CPU_COUNT, random_state=0
            ),
            param_grid=dict(
                n_estimators=[10, 20],
                base_estimator=[
                    None,
                    LogisticRegression(max_iter=2000, n_jobs=CPU_COUNT, random_state=0),
                    SGDClassifier(max_iter=2000, n_jobs=CPU_COUNT, random_state=0),
                ],
            ),
        )),

        ("ensemble", "AdaBoostClassifier", dict(
            estimator=dict(
                random_state=0
            ),
            param_grid=dict(
                n_estimators=[10, 50, 100],
                algorithm=["SAMME", "SAMME.R"],
                learning_rate=[0.5, 1.0],
                base_estimator=[
                    None,
                    LogisticRegression(max_iter=2000, n_jobs=CPU_COUNT, random_state=0),
                    SGDClassifier(max_iter=2000, n_jobs=CPU_COUNT, random_state=0),
                ],
            ),
        )),

        ("svm", "LinearSVC", dict(
            estimator=dict(
                max_iter=2000, random_state=0
            ),
            param_grid=dict(
                penalty=["l1", "l2"],
                loss=["hinge", "squared_hinge"],
                C=[1, 10, 100, 1000],
            ),
        )),

        ("svm", "SVC", dict(
            estimator=dict(
                max_iter=2000, random_state=0
            ),
            param_grid=dict(
                kernel=["linear", "poly", "rbf", "sigmoid"],
                gamma=["scale", "auto", 1, 0.1, 0.001, 0.0001],
                C=[1, 10, 100, 1000],
            ),
        )),

        ("neural_network", "MLPClassifier", dict(
            estimator=dict(
                max_iter=round(10 / train_size), random_state=0
            ),
            param_grid=dict(
                activation=["logistic", "tanh", "relu"],
                solver=["lbfgs", "adam"],
                learning_rate=["constant", "invscaling"],
                alpha=[0.0001, 0.01, 1.0],
            ),
        )),
    ]

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        sentiments,
        train_size=train_size,
        random_state=0
    )

    accuracy, voting, vectorizer, scaler = best_option(
        hashcode, vectorizers, classifiers, x_train, x_test, y_train, y_test
    )

    logging.info("Best accuracy: %.2f%%", accuracy * 100)

    x_train_t, y_train_t, x_test_t, scaler = remove_outliers_and_normalize(
        hashcode,
        vectorizer,
        vectorizer.transform(x_train),
        y_train,
        vectorizer.transform(x_test)
    )

    predictions = voting.predict(x_test_t)

    logging.info("Best accuracy recalculated: %.2f%%", accuracy_score(y_test, predictions) * 100)

    logging.info("Generating most informative features")
    most_informative_features(vectorizer, voting, labels)

    logging.info("Generating plots")
    classifier_plots(x_train_t, y_train_t, x_test_t, y_test, voting, labels)

    logging.info("Running DummyClassifier")
    dummy = DummyClassifier(strategy='most_frequent', random_state=0)
    dummy.fit(x_train_t, y_train_t)
    dummy_score = dummy.score(x_test_t, y_test)

    logging.info("DummyClassifier: %.2f%%", dummy_score * 100)

    test_sentences = [
        ("produto muito bom, adorei! recomendo a todos!", "positive"),
        ("o produto é ótimo, vale a pena comprar!", "positive"),
        ("muito bom, superou minhas expectativas! compraria novamente", "positive"),
        ("a entrega demorou muito e o produto chegou quebrado. muito ruim", "negative"),
        ("terrível, não funciona, tentei de tudo mas não liga", "negative"),
        ("acabamento ruim, muito frágil. não vale a pena.", "negative"),
        ("o produto não é bom como anunciado", "negative"),
    ]

    x_test = [normalize_text(x[0]) for x in test_sentences]
    x_test = [spell_check(x, spell) for x in x_test]
    x_test = [remove_words(x, stop_words) for x in x_test]
    x_test = [stemmed_text(x, stemmer) for x in x_test]
    x_test_t = scaler.transform(vectorizer.transform(x_test))

    predictions = voting.predict(x_test_t)

    for (i, prediction) in enumerate(predictions):
        logging.info("%s | stemmed=%s | predict=%s | real=%s",
                     test_sentences[i][0], x_test[i], sentiments_map[prediction], test_sentences[i][1])


if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"

    # https://github.com/joblib/joblib/issues/138
    # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    multiprocessing.set_start_method("forkserver")

    logging.basicConfig(filename="log.txt", filemode="w", level=logging.INFO,
                        format="%(levelname)s - %(asctime)s - %(message)s")

    nltk.download("stopwords")
    nltk.download("rslp")
    nltk.download("floresta")
    nltk.download("punkt")
    nltk.download("machado")
    nltk.download("mac_morpho")
    spacy.load("pt")

    # https://scikit-learn.org/stable/faq.html#why-do-i-sometime-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux
    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.parallel_backend.html
    with parallel_backend("threading"):
        main()
