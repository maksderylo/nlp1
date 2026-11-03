"""
Shared helpers for our MNLI experiments: sampling by premise, sparse feature matrices,
premise-grouped cross-validation and Bayesian comparison of the results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as st
from joblib import Parallel, delayed
from matplotlib.ticker import PercentFormatter
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from . import analysis

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


def grouped_splits(groups, num_folds=5, seeds=(0, 1, 2)):
    """
    List of (training_ids, validation_ids), num_folds per seed, ordered by seed and then fold.

    The folds are grouped (a premise never appears in both training and validation) and depend only
    on the groups and the seeds, so every system evaluated on the same data gets exactly the same folds.
    """
    groups = np.asarray(groups)
    splits = []
    for seed in seeds:
        folds = GroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits.extend(folds.split(groups, groups=groups))
    return splits


def cross_validate(make_model, X, y, groups, num_folds=5, seeds=(0, 1, 2), n_jobs=-1):
    """
    Train and evaluate a fresh model on every grouped fold, returns the list of validation accuracies.

    make_model: function without arguments that returns a new, untrained model with fit(X, y) and predict(X)
        (e.g. lambda: LogisticRegression(max_iter=1000))
    X: feature matrix (sparse or dense), one row per pair
    y: labels
    groups: promptID per pair
    n_jobs: number of folds trained in parallel (-1: all cores)
    """
    y = np.asarray(y)

    def run_fold(training_ids, validation_ids):
        model = make_model()
        model.fit(X[training_ids], y[training_ids])
        return accuracy_score(y[validation_ids], model.predict(X[validation_ids]))

    splits = grouped_splits(groups, num_folds, seeds)
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
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
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
