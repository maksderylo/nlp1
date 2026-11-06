"""
Shared helpers for our MNLI experiments: sampling by premise, sparse feature matrices,
premise-grouped cross-validation and Bayesian comparison of the results.
"""
import copy
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as st
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import DataLoader

from . import analysis
from . import clslib
from . import smlp

# neutral greys for result plots (in our figures colour is reserved for the three labels)
INK, INK_2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#9c9b95", "#e6e5e1", "#ecebe7"


def sample_by_premise(pairs, num_premises, seed=0):
    """
    Sample premises (promptIDs) and keep all their hypotheses, so the ~3 hypotheses of a premise stay together.

    pairs: DataFrame with a promptID column (e.g. the MNLI training split)
    num_premises: number of premises to sample
    """
    rng = np.random.default_rng(seed)
    prompt_ids = rng.choice(pairs["promptID"].unique(), size=num_premises, replace=False)
    return pairs[pairs["promptID"].isin(prompt_ids)]


def token_matrix(tokenizer, texts, binary=False):
    """
    Sparse matrix with one row per text and one column per token in the tokenizer's vocabulary.

    binary=False: token counts (bag-of-words)
    binary=True: 1 if the token occurs in the text, else 0 (token indicators)
    """
    encodings = tokenizer.encode_batch(list(texts))
    rows = np.repeat(np.arange(len(encodings)), [len(e.ids) for e in encodings])
    cols = np.concatenate([e.ids for e in encodings])
    # repeated (row, col) entries are summed, which gives token counts
    X = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(encodings), tokenizer.get_vocab_size()))
    if binary:
        X.data[:] = 1
    return X


def pair_matrix(tokenizer, premises, hypotheses, binary=False):
    """
    Features for premise-hypothesis pairs: separate blocks for premise and hypothesis (2 x vocab size columns),
    so "not" in the hypothesis is a different feature than "not" in the premise.
    """
    return hstack([
        token_matrix(tokenizer, premises, binary),
        token_matrix(tokenizer, hypotheses, binary),
    ]).tocsr()


NEGATION_WORDS = {"not", "n't", "no", "never", "nobody", "nothing", "none", "nowhere", "neither", "nor", "cannot"}

OVERLAP_STATISTICS = [
    "share of hypothesis words in premise", "new words in hypothesis (log)",
    "premise length (log)", "hypothesis length (log)",
    "negation in premise", "negation in hypothesis", "negation in only one",
]


def overlap_features(tokenizer, premises, hypotheses):
    """
    Features that relate premise and hypothesis to each other, in three sparse blocks:

    shared: bag-of-words of the hypothesis tokens that also occur in the premise (vocab size columns)
    new: bag-of-words of the hypothesis tokens that do not occur in the premise (vocab size columns)
    statistics: one column per name in OVERLAP_STATISTICS. Counts and lengths are log-scaled so all
        values stay small; "words" are tokens with at least one letter or digit (no punctuation).
    """
    vocab = tokenizer.get_vocab()
    # only whole negation words: "▁not" and "n't", but not a word piece "not" inside another word
    negation_ids = {i for token, i in vocab.items() if token == "n't" or (token.startswith("▁") and token[1:] in NEGATION_WORDS)}
    word_ids = {i for token, i in vocab.items() if any(ch.isalnum() for ch in token)}

    shared_rows, shared_cols, new_rows, new_cols, statistics = [], [], [], [], []
    premise_encodings = tokenizer.encode_batch(list(premises))
    hypothesis_encodings = tokenizer.encode_batch(list(hypotheses))
    for row, (premise, hypothesis) in enumerate(zip(premise_encodings, hypothesis_encodings)):
        premise_ids = set(premise.ids)
        for token_id in hypothesis.ids:
            if token_id in premise_ids:
                shared_rows.append(row)
                shared_cols.append(token_id)
            else:
                new_rows.append(row)
                new_cols.append(token_id)

        hypothesis_words = [t for t in hypothesis.ids if t in word_ids]
        num_shared = sum(t in premise_ids for t in hypothesis_words)
        premise_negated = any(t in negation_ids for t in premise.ids)
        hypothesis_negated = any(t in negation_ids for t in hypothesis.ids)
        statistics.append([
            num_shared / max(len(hypothesis_words), 1),
            np.log1p(len(hypothesis_words) - num_shared),
            np.log1p(len(premise.ids)),
            np.log1p(len(hypothesis.ids)),
            premise_negated,
            hypothesis_negated,
            premise_negated != hypothesis_negated,
        ])

    shape = (len(premise_encodings), tokenizer.get_vocab_size())
    shared = csr_matrix((np.ones(len(shared_cols)), (shared_rows, shared_cols)), shape=shape)
    new = csr_matrix((np.ones(len(new_cols)), (new_rows, new_cols)), shape=shape)
    return shared, new, csr_matrix(np.array(statistics, dtype=float))


class MajorityClass:
    """
    Baseline that ignores its input. Its pmf is the relative frequency of each class in the training data,
    and it decides with the tutorial's most-probable-class rule, so it always predicts the most frequent class.
    Has fit/predict, so it can be used with cross_validate like any scikit-learn model.
    """

    def fit(self, X, y):
        counts = np.bincount(np.asarray(y))
        self.pmf = counts / counts.sum()
        return self

    def predict(self, X):
        label = clslib.MostProbableClass()(self.pmf)
        return np.full(X.shape[0], label)


class InteractionMLP(nn.Module):
    """
    MLP that compares premise and hypothesis (in the style of Conneau et al., 2017).

    Input: a sparse batch [premise features | hypothesis features], each block vocab_size wide (as built by pair_matrix),
        optionally followed by num_statistics extra columns (e.g. the overlap statistics) for the hybrid model.
    Process: the same encoder (sparse matmul + ReLU) turns premise and hypothesis into vectors p and h,
        which are combined as [p, h, |p - h|, p * h], followed by a hidden layer (ReLU + dropout) and log-softmax.
        Hybrid (num_statistics > 0): the extra columns are appended unchanged, [p, h, |p - h|, p * h, statistics],
        so the classifier sees the learned comparison and our handcrafted one side by side.
    Output: logprobs over num_classes, like smlp.SparseMLP.
    """

    def __init__(self, vocab_size, hidden_dim, num_classes, dropout=0.2, num_statistics=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_statistics = num_statistics
        # the sentence encoder is shared: premise and hypothesis go through the same weights
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim + num_statistics, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    def encode(self, x):
        return torch.relu(torch.sparse.mm(x, self.weight) + self.bias)

    def forward(self, x):
        # split the sparse batch into its premise columns, its hypothesis columns and (hybrid) its statistics columns
        x = x.coalesce()
        rows, cols = x.indices()
        values = x.values()
        in_premise = cols < self.vocab_size
        in_hypothesis = (cols >= self.vocab_size) & (cols < 2 * self.vocab_size)
        in_statistics = cols >= 2 * self.vocab_size
        shape = (x.shape[0], self.vocab_size)
        premise = torch.sparse_coo_tensor(torch.stack([rows[in_premise], cols[in_premise]]), values[in_premise], shape)
        hypothesis = torch.sparse_coo_tensor(torch.stack([rows[in_hypothesis], cols[in_hypothesis] - self.vocab_size]),
                                             values[in_hypothesis], shape)

        p, h = self.encode(premise), self.encode(hypothesis)
        combined = [p, h, torch.abs(p - h), p * h]
        if self.num_statistics:
            statistics = torch.zeros(x.shape[0], self.num_statistics, dtype=p.dtype)
            statistics[rows[in_statistics], cols[in_statistics] - 2 * self.vocab_size] = values[in_statistics].to(p.dtype)
            combined.append(statistics)
        return self.classifier(torch.cat(combined, dim=-1))


class NeuralClassifier:
    """
    fit/predict wrapper around a torch model, so it can be used with cross_validate like any scikit-learn model.
    Training reuses the tutorial's smlp.train_one_epoch and smlp.evaluate.

    Unlike smlp.train_neural_network_model, which picks the best epoch on the same validation fold that it then
    reports, we hold out part of the *training* fold for early stopping, so the validation fold stays untouched.

    build_model: function input_dim -> torch model returning logprobs (e.g. smlp.SparseMLP)
    batch_size, learning_rate, weight_decay: AdamW settings, defaults as in the tutorial notebook
    early_stopping_share: share of the training fold's premises held out to choose the epoch
    patience: stop after this many epochs without improvement of the held-out loss
    """

    def __init__(self, build_model, batch_size=64, learning_rate=1e-4, weight_decay=1e-4,
                 max_epochs=20, patience=2, early_stopping_share=0.1, seed=0):
        self.build_model = build_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.early_stopping_share = early_stopping_share
        self.seed = seed

    def _loader(self, X, y, shuffle):
        return DataLoader(smlp.SparseDataset(X, y), batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=smlp.sparse_collate)

    @staticmethod
    def _silence_torch_warnings():
        # smlp.sparse_collate triggers two harmless PyTorch performance warnings on every batch
        warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")
        warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

    def fit(self, X, y, groups):
        """groups: promptID per pair, so the held-out part contains whole premises (like our folds)"""
        self._silence_torch_warnings()
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)  # folds already run in parallel, one core each is enough
        y, groups = np.asarray(y), np.asarray(groups)
        # hold out whole premises: if only some hypotheses of a premise were held out, a model that memorises
        # the premise would be pushed away from the held-out label (each premise has one hypothesis per label)
        unique_groups = np.random.default_rng(self.seed).permutation(np.unique(groups))
        held_out_groups = unique_groups[:int(self.early_stopping_share * len(unique_groups))]
        is_held_out = np.isin(groups, held_out_groups)
        held_out, train = np.flatnonzero(is_held_out), np.flatnonzero(~is_held_out)
        train_loader = self._loader(X[train], y[train], shuffle=True)
        held_out_loader = self._loader(X[held_out], y[held_out], shuffle=False)

        self.model = self.build_model(X.shape[1])
        criterion = nn.NLLLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_loss, best_state, epochs_without_improvement = np.inf, None, 0
        self.history = []
        for epoch in range(self.max_epochs):
            train_metrics = smlp.train_one_epoch(self.model, train_loader, optimizer, criterion)
            held_out_metrics, _ = smlp.evaluate(self.model, held_out_loader, criterion)
            self.history.append({"train": train_metrics, "held out": held_out_metrics})
            if held_out_metrics["loss"] < best_loss:
                best_loss, best_state, epochs_without_improvement = held_out_metrics["loss"], copy.deepcopy(self.model.state_dict()), 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    break
        self.model.load_state_dict(best_state)
        self.best_epoch = int(np.argmin([h["held out"]["loss"] for h in self.history])) + 1
        return self

    def predict(self, X):
        self._silence_torch_warnings()
        dummy_labels = np.zeros(X.shape[0], dtype=int)
        _, predictions = smlp.evaluate(self.model, self._loader(X, dummy_labels, shuffle=False), nn.NLLLoss())
        return predictions


def sparse_mlp(input_dim, hidden_dim=128, dropout=0.1, activation="relu", num_hidden_layers=1, num_classes=3):
    """
    The tutorial's smlp.SparseMLP, optionally with tanh instead of ReLU, or with extra hidden layers
    (each extra layer is Linear(hidden_dim, hidden_dim) + activation + dropout, before the output layer).
    With the default arguments it is exactly smlp.SparseMLP(input_dim, 128, 3, dropout=0.1).
    """
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh}
    model = smlp.SparseMLP(input_dim, hidden_dim, num_classes, dropout=dropout)
    model.activation = activations[activation]()
    extra_layers = []
    for _ in range(num_hidden_layers - 1):
        extra_layers += [nn.Linear(hidden_dim, hidden_dim), activations[activation](), nn.Dropout(dropout)]
    model.output = nn.Sequential(*extra_layers, *model.output)
    return model


def grouped_splits(groups, num_folds=5, seeds=(0, 1, 2), grouped=True):
    """
    List of (training_ids, validation_ids), num_folds per seed, ordered by seed and then fold.

    The folds are grouped (a premise never appears in both training and validation) and depend only
    on the groups and the seeds, so every system evaluated on the same data gets exactly the same folds.
    grouped=False: plain k-fold over the pairs instead, ignoring the groups (only for the comparison in the notebook)
    """
    groups = np.asarray(groups)
    splits = []
    for seed in seeds:
        if grouped:
            folds = GroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        else:
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits.extend(folds.split(groups, groups=groups))
    return splits


def cross_validate(make_model, X, y, groups, num_folds=5, seeds=(0, 1, 2), n_jobs=-1, grouped=True):
    """
    Train and evaluate a fresh model on every grouped fold, returns the list of validation accuracies.

    make_model: function without arguments that returns a new, untrained model with fit(X, y) and predict(X)
        (e.g. lambda: LogisticRegression(max_iter=1000)). Models whose fit also takes groups (NeuralClassifier)
        get the promptIDs of the training fold as well.
    X: feature matrix (sparse or dense), one row per pair
    y: labels
    groups: promptID per pair
    n_jobs: number of folds trained in parallel (-1: all cores)
    grouped: False for plain k-fold, where the hypotheses of a premise can end up in different folds
    """
    y, groups = np.asarray(y), np.asarray(groups)

    def run_fold(training_ids, validation_ids):
        model = make_model()
        if isinstance(model, NeuralClassifier):
            model.fit(X[training_ids], y[training_ids], groups=groups[training_ids])
        else:
            model.fit(X[training_ids], y[training_ids])
        return accuracy_score(y[validation_ids], model.predict(X[validation_ids]))

    splits = grouped_splits(groups, num_folds, seeds, grouped)
    return Parallel(n_jobs=n_jobs)(delayed(run_fold)(t, v) for t, v in splits)


def results_frame(accuracies, num_folds=5):
    """
    Turn {system name: list of accuracies} (as returned by cross_validate) into the DataFrame
    expected by analysis.plot_box_and_points, with one row per (system, run, fold).
    """
    rows = [
        (name, accuracy, i // num_folds, i % num_folds)
        for name, accs in accuracies.items()
        for i, accuracy in enumerate(accs)
    ]
    return pd.DataFrame(rows, columns=["model", "accuracy", "run", "fold"])


def compare(accuracies, comparisons, rope, runs=3):
    """
    Bayesian comparison (Benavoli et al., 2017) of pairs of systems, returns a DataFrame with one row per pair.

    accuracies: {system name: list of accuracies}, all evaluated on the same folds
    comparisons: list of (name_a, name_b) pairs
    rope: differences in accuracy smaller than this count as practically equivalent
    runs: number of cross-validation repetitions (seeds)
    """
    rows = []
    for name_a, name_b in comparisons:
        posterior = analysis.BenavoliCorrelatedT(accuracies[name_a], accuracies[name_b], runs=runs, names=[name_a, name_b])
        p_a, p_tie, p_b = posterior.probs(rope)
        # the posterior is a Student-t, so the central 95% interval is also the 95% HDI
        hdi_low, hdi_high = st.t(posterior.df, loc=posterior.mean_diff, scale=posterior.std).interval(0.95)
        rows.append({
            "A": name_a, "B": name_b, "mean diff (B-A)": posterior.mean_diff,
            "95% HDI low": hdi_low, "95% HDI high": hdi_high,
            "P(A wins)": p_a, "P(tie)": p_tie, "P(B wins)": p_b,
        })
    return pd.DataFrame(rows)


def _style_rows(ax, labels):
    """shared look of the row-based result plots: one row per label, first row on top, x grid only"""
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0, labelsize=10, labelcolor=INK)
    ax.tick_params(axis="x", colors=INK_2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ["left", "top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")


def plot_accuracies(accuracies, title="Cross-validated accuracy", reference_lines=None, save_to=None):
    """
    Dot plot of cross-validated accuracies: one row per system, a dot per fold and a bar at the mean.

    accuracies: {system name: list of accuracies}, rows are drawn in this order (top to bottom)
    reference_lines: optional {label: accuracy} drawn as vertical lines, e.g. {"majority class": 0.333}
    save_to: optional path to save the figure (e.g. for the slides)
    """
    names = list(accuracies)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.4))
    row_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    rng = np.random.default_rng(0)  # jitter, so folds with the same accuracy don't hide each other

    for row, name in enumerate(names):
        values = np.asarray(accuracies[name])
        ax.scatter(values, row + rng.uniform(-0.15, 0.15, len(values)), s=30, color=MUTED,
                   edgecolors="white", linewidths=0.8, zorder=2)
        mean = values.mean()
        ax.plot([mean, mean], [row - 0.3, row + 0.3], color=INK, linewidth=2.5, solid_capstyle="round", zorder=3)
        ax.text(1.02, row, f"{mean:.1%}", transform=row_transform, va="center", ha="left", color=INK, fontsize=10)
    ax.text(1.02, -0.5, "mean", transform=row_transform, va="bottom", ha="left", color=INK_2, fontsize=9)

    for label, value in (reference_lines or {}).items():
        ax.axvline(value, color=INK_2, linewidth=1, zorder=1)
        ax.text(value, -0.5, f" {label}", va="bottom", ha="left", color=INK_2, fontsize=9)

    _style_rows(ax, names)
    # decimals only help when all systems are close together
    all_values = np.concatenate([np.asarray(v) for v in accuracies.values()])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1 if np.ptp(all_values) < 0.05 else 0))
    ax.set_xlabel("accuracy (dots: individual folds, bar: mean)", color=INK_2)
    ax.set_title(title, pad=22 if reference_lines else 14)
    if save_to is not None:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.show()


def plot_comparisons(comparison_table, rope, title="Bayesian comparison of accuracies", save_to=None):
    """
    Forest plot of the output of compare(): one row per pair, labelled "B vs. A", with the posterior mean
    difference (B - A), its 95% HDI, the ROPE as a shaded band, and the probabilities that B is worse / tied / better.

    comparison_table: DataFrame returned by compare()
    rope: the ROPE used in compare()
    save_to: optional path to save the figure (e.g. for the slides)
    """
    table = comparison_table.reset_index(drop=True)
    labels = [f"{b}  vs.  {a}" for a, b in zip(table["A"], table["B"])]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(table) + 1.6))
    row_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    # everything in percentage points
    rope_pp = 100 * rope
    ax.axvspan(-rope_pp, rope_pp, color=BAND, linewidth=0, zorder=0)
    ax.axvline(0, color="#c9c8c2", linewidth=1, zorder=1)
    ax.text(0, -0.5, f"ROPE ±{rope_pp:g} pp", va="bottom", ha="center", color=INK_2, fontsize=9)

    for row, r in table.iterrows():
        ax.plot([100 * r["95% HDI low"], 100 * r["95% HDI high"]], [row, row],
                color=INK, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter(100 * r["mean diff (B-A)"], row, s=45, color=INK, edgecolors="white", linewidths=1.5, zorder=3)

    # probabilities as a small table on the right, from the point of view of the first system in the label (B)
    columns = [("P(A wins)", "P(worse)"), ("P(tie)", "P(tie)"), ("P(B wins)", "P(better)")]
    for i, (column, header) in enumerate(columns):
        x = 1.04 + 0.11 * i
        ax.text(x, -0.5, header, transform=row_transform, va="bottom", ha="right", color=INK_2, fontsize=9)
        for row, value in enumerate(table[column]):
            ax.text(x, row, f"{value:.1%}", transform=row_transform, va="center", ha="right", color=INK, fontsize=10)

    limit = 1.3 * max(100 * table[["95% HDI low", "95% HDI high"]].abs().max().max(), rope_pp)
    ax.set_xlim(-limit, limit)
    _style_rows(ax, labels)
    ax.set_xlabel("difference in accuracy, first − second system (percentage points; dot: posterior mean, line: 95% HDI)", color=INK_2)
    ax.set_title(title, pad=22)
    if save_to is not None:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.show()


def accuracy_interval(accuracy, num_examples, z=1.96):
    """95% interval (normal approximation) for an accuracy measured once on num_examples examples"""
    half_width = z * np.sqrt(accuracy * (1 - accuracy) / num_examples)
    return accuracy - half_width, accuracy + half_width


def plot_dev_accuracies(dev_results, num_examples, title="Accuracy on the dev sets", save_to=None):
    """
    Dot plot of held-out accuracies: one row per system, a filled dot for dev matched and an open dot for
    dev mismatched, each with its 95% interval.

    dev_results: DataFrame with one row per system and the columns "dev matched" and "dev mismatched"
    num_examples: {"dev matched": n, "dev mismatched": n}, used for the intervals
    """
    names = list(dev_results.index)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(names) + 1.6))
    styles = {"dev matched": dict(color=INK, offset=-0.13), "dev mismatched": dict(color="white", offset=0.13)}

    for split, style in styles.items():
        for row, accuracy in enumerate(dev_results[split]):
            low, high = accuracy_interval(accuracy, num_examples[split])
            y = row + style["offset"]
            ax.plot([low, high], [y, y], color=INK_2, linewidth=1.5, solid_capstyle="round", zorder=2)
            ax.scatter(accuracy, y, s=42, color=style["color"], edgecolors=INK, linewidths=1.5, zorder=3)

    _style_rows(ax, names)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("accuracy (line: 95% interval)", color=INK_2)
    legend = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=s["color"], markeredgecolor=INK,
                     markeredgewidth=1.5, label=split) for split, s in styles.items()]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9,
              borderaxespad=0.2)
    ax.set_title(title, pad=26)
    if save_to is not None:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(predictions, gold, labels, title="Confusion matrices", save_to=None):
    """
    One confusion matrix per system, side by side. Each row is a gold label and is normalised to 100%,
    so a cell says: of the pairs with this gold label, which share got this prediction.

    predictions: {system name: predicted label ids}
    gold: gold label ids
    labels: label names, in the order of the label ids
    """
    cmap = LinearSegmentedColormap.from_list("ink", ["#fcfcfb", "#2b2a27"])  # one hue, light to dark
    fig, axes = plt.subplots(1, len(predictions), figsize=(3.6 * len(predictions), 3.4), squeeze=False)
    for ax, (name, predicted) in zip(axes[0], predictions.items()):
        matrix = confusion_matrix(gold, predicted, labels=range(len(labels)), normalize="true")
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if matrix[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted", color=INK_2)
        ax.set_ylabel("gold", color=INK_2)
        ax.set_title(name, fontsize=10)
        ax.grid(False)
        ax.tick_params(length=0, colors=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=200, bbox_inches="tight")
    plt.show()
