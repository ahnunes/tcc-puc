#!/usr/bin/env python

"""
Auxiliary functions
"""

import copy
import hashlib
import html
import importlib
import inspect
import io
import logging
import os
import re
import unicodedata
from functools import partial
from itertools import combinations
from time import time

import numpy as np
import pandas as pd
import spacy
import unidecode

import catboost

import joblib
from joblib import Parallel, delayed, parallel_backend

import h5py
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import PercentFormatter

from mpl_toolkits.axes_grid1 import make_axes_locatable

import nltk
from nltk.corpus import stopwords, floresta, machado, mac_morpho
from nltk.stem import RSLPStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from symspellpy.symspellpy import SymSpell

from toolz import compose

from wordcloud import WordCloud

from leia import SentimentIntensityAnalyzer

CPU_COUNT = os.cpu_count()
REMOVED_STOP_WORDS = dict()


class CatBoostClassifier(catboost.CatBoostClassifier):
    """Implement __repr__"""

    def __repr__(self):
        return f"CatBoostClassifier(sk_params={self._init_params})"


class KerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    _estimator_type = 'classifier'

    def __repr__(self):
        return f"KerasClassifier(sk_params={self.sk_params})"

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state  # pylint: disable=attribute-defined-outside-init


class SkipGramVectorizer(CountVectorizer):
    """https://stackoverflow.com/questions/39725052/implementing-skip-gram-with-scikit-learn"""

    def __init__(self, k=1, **kwds):
        super(SkipGramVectorizer, self).__init__(**kwds)
        self.k = k
        self.kwds = kwds

    def __repr__(self, N_CHAR_MAX=700):
        return f"SkipGramVectorizer(k={self.k}, kwds={self.kwds})"

    def build_sent_analyzer(self, preprocess, stop_words, tokenize):
        """Return sentence analyzer lambda"""
        return lambda sent: self._word_skip_grams(
            compose(tokenize, preprocess, self.decode)(sent),
            stop_words)

    def build_analyzer(self):
        """Return analyzer lambda"""
        preprocess = self.build_preprocessor()
        stop_words = self.get_stop_words()
        tokenize = self.build_tokenizer()
        sent_analyze = self.build_sent_analyzer(preprocess, stop_words, tokenize)

        return sent_analyze

    def _word_skip_grams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        k = self.k
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    # k-skip-n-grams
                    head = [original_tokens[i]]
                    for skip_tail in combinations(original_tokens[i+1:i+n+k], n-1):
                        tokens_append(space_join(head + list(skip_tail)))

        return tokens


def flatten_list(l):
    """Deep list flatten"""

    for item in l:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item


def normalize_text(text):
    """Normalize text"""

    if text is None:
        return ""

    # Convert to lower case
    text = text.lower()

    # Unescape HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^.]*>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
        " ", text)

    # Remove the whole word if it has numbers
    text = re.sub(r"\b[0-9a-z]*[0-9]+[0-9a-z]*\b", " ", text)

    # Remove accents and non-alphabetic characters
    text = unidecode.unidecode(text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[^a-z ]", " ", text)

    # Remove double spaces
    text = re.sub(r"\s\s+", " ", text)

    # Remove double vogals
    text = re.sub(r"([aeiou])\1+", r"\1", text)

    # Remove words with less than 3 characters
    text = re.sub(r"\b[a-z][a-z]?\b", "", text)

    # Remove spaces from begin and end of the string
    text = text.strip()

    return text


def remove_words(text, words):
    """Remove specified words from given text"""

    text_split = text.split()

    for x in text_split:
        if x in words:
            REMOVED_STOP_WORDS[x] = REMOVED_STOP_WORDS.get(x, 0) + 1

    text = " ".join([x for x in text_split if x not in words])

    return text


def stemmed_text(text, stemmer=None):
    """Stemm text"""
    if stemmer is None:
        stemmer = RSLPStemmer()

    return " ".join([stemmer.stem(x) for x in text.split()])


def sentiment(rating, source):
    """Return sentiment for a given rating"""

    if source == "ReclameAqui":
        rating /= 2

    if rating < 3:
        return "negative"
    elif rating >= 3:
        return "positive"


def bar_autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    """

    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")


def leia_sentiment(text, analyzer=None):
    """Return Leia sentiment"""
    if analyzer is None:
        analyzer = SentimentIntensityAnalyzer()

    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0:
        return "positive"
    else:
        return "negative"


def spell_check(text, spell):
    """Fix misspelled words"""
    suggestions = spell.lookup_compound(text, 2)

    if suggestions:
        return suggestions[0].term
    else:
        return text


def load_spell_checker():
    """Return spell checker"""
    if not os.path.exists("data/unigrams.txt"):
        sents = [normalize_text(" ".join(x)).split() for x in floresta.sents()]
        sents += [normalize_text(" ".join(x)).split() for x in machado.sents()]
        sents += [normalize_text(" ".join(x)).split() for x in mac_morpho.sents()]

        unigrams = [item for sublist in sents for item in sublist]
        unigrams = nltk.probability.FreqDist(unigrams)

        file = open("data/unigrams.txt", "w")
        for k, v in unigrams.items():
            file.write(f"{k} {v}\n")
        file.close()

        bigrams = []

        for sent in sents:
            bigrams += list(nltk.bigrams(sent))

        bigrams = nltk.probability.FreqDist(bigrams)

        file = open("data/bigrams.txt", "w")
        for k, v in bigrams.items():
            file.write(f"{' '.join(k)} {v}\n")
        file.close()

    result = SymSpell()

    result.load_dictionary("data/unigrams.txt", 0, 1)
    result.load_bigram_dictionary("data/bigrams.txt", 0, 2)

    return result


def df_parallel_apply(df, func, series=CPU_COUNT*CPU_COUNT, backend="loky"):
    """Apply function in parallel"""

    chunks = np.array_split(df, series)
    print(f"Executing {len(chunks)} chunks in parallel")

    def execute(chunk, func):
        return chunk.apply(func, axis=1)

    with parallel_backend(backend):
        result = Parallel(n_jobs=CPU_COUNT, verbose=10)(delayed(execute)(x, func) for x in chunks)

    result = [item for sublist in result for item in sublist]

    return result


def get_stop_words():
    """Return stop words set"""
    result = stopwords.words("portuguese") + list(spacy.lang.pt.stop_words.STOP_WORDS)
    result = [normalize_text(x) for x in result]

    # https://gist.githubusercontent.com/alopes/5358189/raw/2107d809cca6b83ce3d8e04dbd9463283025284f/stopwords.txt
    # https://raw.githubusercontent.com/stopwords-iso/stopwords-pt/master/stopwords-pt.txt
    with open("data/stop-words.txt", "r") as reader:
        result += [normalize_text(x) for x in reader.read().splitlines()]

    result = set(result)

    booster = open("lexicons/booster.txt", "r")
    mandatory = set(booster.read().split())
    booster.close()

    mandatory.discard("INCR")

    mandatory = mandatory.union(mandatory, {
        "nao",
        "boa",
        "bom",
        "bem",
        "ruim",
        "muita",
        "pouca",
        "pouco",
        "maximo",
        "tanta",
        "nunca",
        "falta",
        "nenhuma",
        "obrigado",
        "obrigada",
    })

    result = [x for x in result if x not in mandatory]

    return result


def language(text, wtl):
    """Return language from given text and using the provided language classifier and regex"""

    words = [
        "recomendo", "amei", "entrega", "otim[ao]", "excelente", "rapida", "celular", "gostei",
        "facil", "lindo", "bonito", "comprei", "legal", "perfume", "preco", "tela", "pra", "lento",
        "problema", "pelicula", "memoria", "cabelo", "ultima",
    ]

    if re.search(rf'\b({"|".join(words)})\b', text):
        result = "pt"
    else:
        result = wtl.predict_lang(text)

    return result


def time_format(time_int, decimals=1):
    """Convert time differences into string using biggest unit"""
    if np.isnan(time_int):
        return "-"
    elif time_int > 3600:
        return "{:.{dec}f}h".format(time_int / 3600, dec=decimals)
    elif time_int > 60:
        return "{:.{dec}f}m".format(time_int / 60, dec=decimals)
    elif time_int >= 1:
        return "{:.{dec}f}s".format(time_int, dec=decimals)

    return "{:.0f}ms".format(time_int * 1000)


def cached_instance(hashcode, name, values, x, y=None, family=None, grid_search=False):
    """Return stored vectorizer or instantiate a brand new one"""

    if grid_search:
        try:
            classifier_class = getattr(importlib.import_module(family), name)
        except Exception:  # pylint: disable=broad-except
            classifier_class = getattr(importlib.import_module(f"sklearn.{family}"), name)

        all_parameters = inspect.signature(classifier_class).parameters

        if 'verbose' in all_parameters or name == "KerasClassifier":
            values["estimator"]["verbose"] = 0

        values["estimator"] = classifier_class(**values["estimator"])
        family, name = "model_selection", "GridSearchCV"
    elif isinstance(values.get("estimator", None), dict):
        values = values["estimator"]

    if family is None:
        if name == "SkipGramVectorizer":
            family = "aux"
        elif "Vectorizer" in name:
            family = "feature_extraction.text"

    # Ignore in the hash, since they don't impact the classifier
    hash_values = values.copy()
    hash_values.pop("verbose", None)
    hash_values.pop("n_jobs", None)
    hash_values.pop("build_fn", None)

    hashcode += str((hash_values))

    if hasattr(x, "shape") and len(x.shape) > 1:
        hashcode += str((x.shape, x[:10].A[0]))
    else:
        hashcode += str((len(x), x[:10]))

    hashcode = hashlib.blake2s(hashcode.encode("utf-8"), digest_size=8).hexdigest()

    filename = f"cache/{name}_{hashcode}.jbl"

    if os.path.exists(filename):
        return joblib.load(filename)

    try:
        _class = getattr(importlib.import_module(family), name)
    except Exception:  # pylint: disable=broad-except
        _class = getattr(importlib.import_module(f"sklearn.{family}"), name)

    all_parameters = inspect.signature(_class).parameters

    if 'verbose' in all_parameters:
        values["verbose"] = values.get("verbose", 10 if grid_search else 1)

    if 'n_jobs' in all_parameters:
        values["n_jobs"] = values.get("n_jobs", CPU_COUNT)

    instance = _class(**values)

    t0 = time()
    instance.fit(x, y)
    fit_time = time() - t0

    logging.info("Fit time: %s", time_format(fit_time, decimals=3))

    result = (instance, fit_time)

    joblib.dump(result, filename)
    logging.info("Saved '%s'", filename)

    return result


def wordcloud_plot(df, suffix):
    """Generate wordcloud plots"""
    plots = [
        (f"plot/wordcloud_{suffix}_all.png", " ".join(df.text)),
        (f"plot/wordcloud_{suffix}_positive.png", " ".join(df[df["sentiment"] == "positive"].text)),
        (f"plot/wordcloud_{suffix}_negative.png", " ".join(df[df["sentiment"] == "negative"].text)),
    ]

    for filename, text in plots:
        wordcloud = WordCloud(background_color="white", collocations=False, width=1000, height=600).generate(text)

        plt.subplots(1, 1, figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(filename)
        logging.info("Saved '%s'", filename)


def data_plots(df):
    """Generate plots for the given data frame"""

    logging.info("Generating plots")

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Plot with reviews count grouped by source
    by_source = df.groupby("source")
    counts = df.source.value_counts()
    total = counts.sum()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    counts.plot(kind="pie", labeldistance=None, legend=True)
    ax.set_ylabel("")
    h, l = ax.get_legend_handles_labels()
    l = [f"{x} ({counts[x] / total:.1%})" for x in l]
    ax.legend(h, l, loc="upper left", bbox_to_anchor=(1.0, 0.9), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("plot/by_source.png")
    logging.info("Saved 'plot/by_source.png'")

    # Plot with reviews average length grouped by source
    fig, ax = plt.subplots()
    by_source["length"].mean().plot(kind="bar", color=colors)
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#dddddd")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig("plot/by_length1.png")
    logging.info("Saved 'plot/by_length1.png'")

    # Plot with reviews average length grouped by sentiment
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    colors = ["green", "red"]
    df.groupby("sentiment")["length"].mean().sort_index(ascending=False).plot(kind="bar", color=colors)
    ax.set_xlabel("")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#dddddd")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig("plot/by_length2.png")
    logging.info("Saved 'plot/by_length2.png'")

    # Plot with reviews count by sentiment
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    by_sentiment = df.sentiment.value_counts()
    sentiments = ["positive", "negative"]
    x = np.arange(len(sentiments))
    width = 0.6

    for (i, label) in enumerate(sentiments):
        points = by_sentiment[by_sentiment.index == label]
        rects = ax.bar([i], points, width, label=label, color=colors[i])
        bar_autolabel(rects, ax)

    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(sentiments)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#dddddd")
    ax.legend()

    plt.ylim(bottom=0, top=max(by_sentiment) * 1.1)
    fig.tight_layout()
    plt.savefig("plot/by_sentiment1.png")
    logging.info("Saved 'plot/by_sentiment1.png'")

    # Plot with reviews count by source and sentiment
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    by_sentiment = df.groupby(["source", "sentiment"])["sentiment"].count()
    sources = df.source.value_counts().index
    x = np.arange(len(sources))
    width = 0.9 / len(sentiments)

    for (i, label) in enumerate(sentiments):
        points = by_sentiment[by_sentiment.index.get_level_values(1) == label]
        points = [points[points.index.get_level_values(0) == x].get(0, 0) for x in sources]

        rects = ax.bar(x + i * width, points, width, label=label, color=colors[i])
        bar_autolabel(rects, ax)

    ax.set_ylabel("Count")
    ax.set_xticks(x + width * (len(sentiments) - 1) / 2)
    ax.set_xticklabels(sources)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#dddddd")
    ax.legend()

    plt.ylim(bottom=0, top=max(by_sentiment) * 1.1)
    fig.tight_layout()
    fig.savefig("plot/by_sentiment2.png")
    logging.info("Saved 'plot/by_sentiment2.png'")


def combined_score(combination, predictions, y_test):
    """
    Combine several predictions by majority
    Based on https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4
    """
    predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)),
                                      arr=np.asarray(predictions).T.astype("int"),
                                      axis=1)

    score = accuracy_score(y_test, predictions)
    print(f"{score:.2%} {combination}")

    return score


def group_df(df, average=True, group_by="accuracy", ascending=False):
    """Return grouped data frame by vectorizers and classifiers"""

    df_grouped = pd.crosstab(
        df.c_name,
        df.v_name,
        values=df[group_by],
        aggfunc="mean",
        margins=True,
        margins_name="Average",
    )

    v_names = list(pd.unique(df["v_name"]))

    df_grouped = df_grouped.sort_values(by=["Average"], ascending=ascending)

    if average:
        v_names.append("Average")
    else:
        del df_grouped["Average"]
        df_grouped = df_grouped[df_grouped.index != "Average"]

    df_grouped = df_grouped[[x for x in v_names if x in df_grouped.columns]]

    if group_by == "accuracy":
        logging.info(df_grouped)

    return df_grouped


def plot_lines(ax, v=None, h=None, color=None):
    """Plot vertical and horizontal lines"""
    if v is not None:
        for x in v:
            ax.axvline(x, color=color)

    if h is not None:
        for x in h:
            ax.axhline(x, color=color)


def plot_box(ax, left=0, right=1, bottom=0, top=1, color=None, zorder=1000, shadow=True):
    """Plot box"""
    ax.vlines(left, bottom, top, color=color, zorder=zorder)
    ax.vlines(right, bottom, top, color=color, zorder=zorder)
    ax.hlines(bottom, left, right, color=color, zorder=zorder)
    ax.hlines(top, left, right, color=color, zorder=zorder)

    if shadow:
        ax.fill_between([left, right], bottom, top, facecolor=color, alpha=0.1, zorder=zorder)


def heatmap_plot(df, filename=None, average=True, group_by="accuracy", str_format="{:.2%}", box="max", print_flag=True, time_heatmap=False):
    """Generate heatmap plot"""

    df_grouped = group_df(df, average=average, group_by=group_by, ascending=box == "min")
    v_names = list(df_grouped.columns)

    if average:
        avg = df_grouped[df_grouped.index == "Average"].iloc[0]
        v_names.sort(key=lambda x: round(avg[x], 4) * (1 if box == "max" else -1))
        df_grouped = df_grouped[v_names]

    c_names = list(df_grouped.index)
    min_value = min(df[group_by])
    max_value = max(df[group_by])
    ratio = (len(c_names) / len(v_names)) / 2

    figsize = (len(v_names) * 0.8 + 3.1, 2 + ratio * len(v_names) * 0.8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.subplots_adjust(wspace=0.2)
    plt.title(group_by)

    cmap = cm.get_cmap("Greens", 512)
    cb_cmap = im_cmap = ListedColormap(cmap(np.linspace(0.0, 0.5, 256)))

    if box == "min":
        im_cmap = ListedColormap(cmap(np.linspace(0.5, 0.0, 256)))

    ax.imshow(df_grouped, cmap=im_cmap, aspect="auto")

    if isinstance(str_format, str):
        str_formater = str_format.format
    else:
        str_formater = str_format

    # Create colorbar
    if min_value != max_value:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.2)

        ticks = np.linspace(min_value, max_value, 10)

        if box == "min":
            ticks = ticks[::-1]

        norm = Normalize(vmin=0, vmax=9, clip=False)
        cbar = ColorbarBase(cax, cmap=cb_cmap, ticks=range(10), norm=norm)
        cbar.ax.set_yticklabels([str_formater(ticks[i]) for i in cbar.get_ticks()])

    xticks = np.arange(len(v_names))
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(len(c_names)))
    ax.set_yticklabels([x.split(".")[-1] for x in c_names])
    ax.set_xticklabels(v_names)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for (i, c_name) in enumerate(c_names):
        for (j, v_name) in enumerate(v_names):
            value = df_grouped[df_grouped.index == c_name][v_name].values[0]
            value = str_formater(value)
            ax.text(j, i, value, ha="center", va="center", color="black")

            if print_flag and "flag" in df.columns and "Average" not in [c_name, v_name]:
                row = df[(df.c_name == c_name) & (df.v_name == v_name)]

                if row.empty:
                    continue

                flag = row.iloc[0].flag

                if flag:
                    ax.text(j + 0.4, i - 0.25, flag, ha="center", va="center", color="black")

    ax.set_xticks(np.arange(len(v_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(c_names)) - 0.5, minor=True)
    ax.grid(True, which="minor", linestyle="-", color="w")
    ax.tick_params(which="minor", length=0.0)

    if average:
        x_avg = v_names.index("Average")
        y_avg = c_names.index("Average")
        plot_lines(ax, v=[x_avg - 0.5, x_avg + 0.5], h=[y_avg - 0.5, y_avg + 0.5], color="red")

    box_value = max(df[group_by]) if box == "max" else min(df[group_by])
    box_row = df[df[group_by] == box_value].iloc[0]
    x, y = v_names.index(box_row.v_name), c_names.index(box_row.c_name)
    plot_box(ax, left=x-0.5, right=x+0.5, bottom=y-0.5, top=y+0.5, color="b")

    fig.tight_layout()

    if filename is None:
        filename = f"{len(v_names)}_{len(c_names)}"

    plt.savefig(f"plot/heatmap_{filename}.png")
    logging.info("Saved 'plot/heatmap_%s.png'", filename)

    if average:
        ax.fill_between(np.arange(-0.5, x_avg + 1, 0.5), -0.5, len(c_names) - 0.5, facecolor="black", alpha=0.2)
        ax.fill_between(np.arange(x_avg + 0.5, len(v_names), 0.5), y_avg -
                        0.5, len(c_names) - 0.5, facecolor="black", alpha=0.2)

        plt.savefig(f"plot/heatmap_{filename}_2.png")
        logging.info("Saved 'plot/heatmap_%s_2.png'", filename)

    if group_by == "accuracy" and time_heatmap:
        df["total_time"] = df.apply(lambda row: row.train_time + row.predict_time, axis=1)

        heatmap_plot(df, filename=f"{filename}_train_time", average=average, print_flag=False,
                     group_by="train_time", str_format=partial(time_format, decimals=2), box="min")

        heatmap_plot(df, filename=f"{filename}_predict_time", average=average, print_flag=False,
                     group_by="predict_time", str_format=partial(time_format, decimals=3), box="min")

        heatmap_plot(df, filename=f"{filename}_total_time", average=average, print_flag=False,
                     group_by="total_time", str_format=partial(time_format, decimals=2), box="min")


def most_informative_features(vectorizer, classifier, class_labels):
    """
    Prints features with the highest coefficient values, per class
    https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
    """
    feature_names = vectorizer.get_feature_names()

    for estimator in classifier.estimators_:
        if hasattr(estimator, "coef_"):
            if len(class_labels) > 2:
                for i, class_label in enumerate(class_labels):
                    top = np.argsort(estimator.coef_[i])[-10:]
                    logging.info("%s | %s: %s", estimator.__class__.__name__, class_label, ", ".join(feature_names[j] for j in top))
            else:
                top = np.argsort(estimator.coef_[0])

                logging.info("%s | %s: %s", estimator.__class__.__name__, class_labels[0], ", ".join(feature_names[j] for j in top[:10]))
                logging.info("%s | %s: %s", estimator.__class__.__name__, class_labels[1], ", ".join(feature_names[j] for j in top[-10:]))


def plot_learning_curves(x_train, y_train, x_test, y_test, clf):
    """
    Plots learning curves of a classifier.

    Based on mlxtend:
    https://github.com/rasbt/mlxtend/blob/master/mlxtend/plotting/learning_curves.py
    """
    errors = []

    def misclf_err(y_predict, y):
        return (y_predict != y).sum() / float(len(y))

    rng = [int(i) for i in np.linspace(0, x_train.shape[0], 11)][1:]
    for r in rng:
        print(f"Running {r / x_train.shape[0]:.0%}")
        clf.fit(x_train[:r], y_train[:r])

        y_test_predict = clf.predict(x_test)

        error = misclf_err(y_test, y_test_predict)
        errors.append(error*100)

    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(np.arange(10, 101, 10), errors, label="Error", marker="o")

    ax.set_xlim([0, 110])
    ax.set_xticks(np.arange(0, 101, 10))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.xaxis.set_major_formatter(PercentFormatter())

    plt.ylabel("Error")
    plt.xlabel("Training set size in percent")
    plt.title("Learning Curve")
    plt.grid()
    plt.tight_layout()


def plot_precision_recall_curve(x_train_t, y_train_t, x_test_t, y_test, classifier, class_labels):
    """Plot Precision-Recall curve"""
    all_curves = []

    for estimator in classifier.estimators_:
        if not hasattr(estimator, "decision_function"):
            continue

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        clf = OneVsRestClassifier(estimator, n_jobs=CPU_COUNT)
        y_score = clf.fit(x_train_t, y_train_t).decision_function(x_test_t)

        binary = len(classifier.classes_) == 2

        if not binary:
            for i in range(len(classifier.classes_)):
                precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
                ax.plot(recall * 100, precision * 100, label=class_labels[i])

        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
        ax.plot(recall * 100, precision * 100, label="all")

        all_curves.append((estimator.__class__.__name__, precision, recall))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.set_xticks(np.arange(0, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.xaxis.set_major_formatter(PercentFormatter())

        if not binary:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        plt.grid()
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title(f"Precision-Recall Curve for {estimator.__class__.__name__}")
        plt.savefig(f"plot/precision_recall_{estimator.__class__.__name__}.png")
        logging.info("Saved 'plot/precision_recall_%s.png'", estimator.__class__.__name__)

    if all_curves:
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        for label, precision, recall in all_curves:
            ax.plot(recall * 100, precision * 100, label=label)

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

        ax.set_xticks(np.arange(0, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.grid()
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.title(f"Precision-Recall Curve")
        plt.savefig(f"plot/precision_recall_all.png")
        logging.info("Saved 'plot/precision_recall_all.png'")


def plot_roc_curve(x_train_t, y_train_t, x_test_t, y_test, classifier, class_labels):
    """
    Based on:
    https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    """

    all_curves = []

    for estimator in classifier.estimators_:
        if not hasattr(estimator, "decision_function"):
            continue

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        clf = OneVsRestClassifier(estimator, n_jobs=CPU_COUNT)
        y_score = clf.fit(x_train_t, y_train_t).decision_function(x_test_t)

        binary = len(classifier.classes_) == 2

        if not binary:
            for i in range(len(classifier.classes_)):
                fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr * 100, tpr * 100, label=f"{class_labels[i]}\n{roc_auc:.2%}")

        fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr * 100, tpr * 100, label=f"all\n{roc_auc:.2%}")
        ax.plot([0, 100], [0, 100], color='navy', linestyle='--')

        all_curves.append((estimator.__class__.__name__, fpr, tpr, roc_auc))

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(np.arange(0, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.xaxis.set_major_formatter(PercentFormatter())

        if not binary:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.title(f"Receiver Operating Characteristic Curve for {estimator.__class__.__name__}")
        plt.savefig(f"plot/roc_{estimator.__class__.__name__}.png")
        logging.info("Saved 'plot/roc_%s.png'", estimator.__class__.__name__)

    if all_curves:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        for label, fpr, tpr, roc_auc in all_curves:
            ax.plot(fpr * 100, tpr * 100, label=f"{label}\n{roc_auc:.2%}")

        ax.plot([0, 100], [0, 100], color='navy', linestyle='--')

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

        ax.set_yticks(np.arange(0, 101, 10))
        ax.set_xticks(np.arange(0, 101, 10))
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid()
        plt.title(f"Receiver Operating Characteristic Curve")
        plt.savefig(f"plot/roc_all.png")
        logging.info("Saved 'plot/roc_all.png'")


def classifier_plots(x_train_t, y_train_t, x_test_t, y_test, classifier, class_labels):
    """Plot information related to the classifier"""
    plt.clf()

    y_train_b = label_binarize(y_train_t, classes=classifier.classes_)
    y_test_b = label_binarize(y_test, classes=classifier.classes_)

    logging.info("Confusion Matrix")

    plot_confusion_matrix(classifier, x_test_t, y_test, display_labels=class_labels, values_format="d")
    plt.gca().invert_yaxis()
    plt.title("Confusion Matrix")
    plt.savefig("plot/confusion_matrix.png")
    logging.info("Saved 'plot/confusion_matrix.png'")

    plot_confusion_matrix(classifier, x_test_t, y_test, display_labels=class_labels, normalize="all")
    plt.title("Confusion Matrix")
    plt.savefig("plot/confusion_matrix_normalized.png")
    logging.info("Saved 'plot/confusion_matrix_normalized.png'")

    logging.info("ROC curve")
    plot_roc_curve(x_train_t, y_train_b, x_test_t, y_test_b, classifier, class_labels)

    logging.info("Precision-Recall curve")
    plot_precision_recall_curve(x_train_t, y_train_b, x_test_t, y_test_b, classifier, class_labels)

    logging.info("Learning curve")

    plot_learning_curves(x_train_t, y_train_t, x_test_t, y_test, classifier)
    plt.savefig("plot/learning_curve.png")
    logging.info("Saved 'plot/learning_curve.png'")
