import numpy as np
import itertools
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from sklearn.metrics import accuracy_score
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, OrderedDict
from tqdm.auto import tqdm
import random

from . import loglin
from . import smlp
from . import clslib


def download_ag_news_texts():
    """
    Returns the texts in the AG news collection (but not the labels).
    """
    # Loading our training corpus
    dataset = load_dataset("fancyzhx/ag_news")
    corpus = dataset["train"]["text"]

    return corpus


def train_tokenizer(
    corpus,
    vocab_size,
    min_frequency,
    pretokenizer=Metaspace(),
    unk_token="[UNK]",
    normalizer=None,
    show_progress=True,
):
    """
    corpus: a list of texts (each a string)
    vocab_size: number of distinct BPE tokens
    min_frequency: we only merge token pairs that are frequent enough
    pretokenizer: a simple tokenizer that we use before BPE tokenization

        You probably don't need to change the pretokenizer, unless you are working with
        languages that really don't use any form of blank space
        (then you could, for example, pretokenize at character level)
    normalizer: optional normalisation applied before pretokenization (e.g. tokenizers.normalizers.Lowercase())
    show_progress: show the trainer's progress bar (in some notebook frontends it shows up as blank lines)
    """
    # Create an empty BPE tokenizer
    tokenizer = Tokenizer(
        BPE(unk_token=unk_token)
    )  # unk_token used for characters that cannot be encoded

    # Optionally normalise the text (e.g. lowercase) before anything else
    if normalizer is not None:
        tokenizer.normalizer = normalizer

    # First split the corpus into words
    tokenizer.pre_tokenizer = pretokenizer

    # Tell BPE how large the vocabulary should be
    trainer = BpeTrainer(
        vocab_size=vocab_size,  # The size of the vocabulary to learn.
        min_frequency=min_frequency,  # Only perform a merge if the pair occurs at least min_frequency times in the training corpus.
        special_tokens=[unk_token],
        show_progress=show_progress,
    )

    # Learn the vocabulary and merge rules
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    return tokenizer


def split_test_devtest(pairs, devtest_size, seed=0):
    """
    Shuffles a list of labelled texts and splits the result into a devtest set of a given size
    and a test set with what's left.

    pairs: lists of tuples (text, label)
    devtest_size: size of the devtest split
    seed: for reproducible shuffling
    """
    pairs = list(pairs)
    ids = np.random.default_rng(seed=seed).permutation(np.arange(len(pairs)))
    split = {
        "devtest": [pairs[i] for i in ids[:devtest_size]],
        "devtest_ids": [i for i in ids[:devtest_size]],
        "test": [pairs[i] for i in ids[devtest_size:]],
        "test_ids": [i for i in ids[devtest_size:]],
    }
    return split


def cross_validation_split(pairs, k, seed=0):
    """
    Shuffles a list of labelled texts and splits the result into k folds.

    pairs: lists of tuples (text, label)
    k: number of folds
    seed: for reproducible shuffling
    """

    # Shuffle the pairs
    random.seed(seed)
    pairs = list(pairs)
    ids = list(range(len(pairs)))
    random.shuffle(ids)

    # Split without coercing (text, int) tuples into a string array
    ids_parts = np.array_split(np.array(ids, dtype=object), k)

    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]]
        training_ids = list(
            itertools.chain.from_iterable(
                part for k, part in enumerate(ids_parts) if k != i
            )
        )
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set,
            "validation": validation_set,
            "training_ids": training_ids,  # these are not often needed, but we return it just in case
            "validation_ids": ids_parts[
                i
            ],  # these are not often needed, but we return it just in case
        }

    return cross_validation_sets


def cross_validation_experiment(
    data_splits,
    outcome_space,
    encoder: "Encoder",
    model_option: str,
    save_to="tmp",
    loglin_args=dict(),
    smlp_args=dict(),
):
    """
    Trains and evaluates k models using a k-fold data split for cross validation.
    This function returns a dictionary with
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition

    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()

    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True)

        if model_option == "logistic":
            llmodel = loglin.fit(
                training_set, encoder, save_to=f"{fold_folder}/model", **loglin_args
            )
            models[fold] = f"{fold_folder}/model"

            predictions[fold] = loglin.predict(
                [text for text, gold_label in validation_set], llmodel, encoder
            )
            gold_labels_validation = [gold_label for text, gold_label in validation_set]
            accuracies[fold] = accuracy_score(gold_labels_validation, predictions[fold])
        elif model_option == "mlp":
            metrics, best_epoch, predictions[fold] = smlp.train_neural_network_model(
                training_set,
                validation_set,
                len(outcome_space),
                encoder,
                save_to=f"{fold_folder}/model",
                **smlp_args,
            )
            models[fold] = f"{fold_folder}/model"
            accuracies[fold] = metrics[best_epoch]["val_metrics"]["accuracy"]
        else:
            raise ValueError(f"Unkonwn model_option={model_option}")

    return {"models": models, "predictions": predictions, "accuracies": accuracies}


class WrapperForLoglinearModel(clslib.ProbabilisticModel):
    """
    Wrapper to treat an instance of sklearn LogisticRegression in the API of a ProbabilisticModel
    """

    def __init__(self, encoder, model):
        """
        text encoder used for training and the trained model
        """
        self.encoder = encoder
        self.model = model

    def __call__(self, x):
        """Predicts a pmf for x"""
        r = loglin.predict_probs([x], self.model, self.encoder)
        return r[0]


def ensemble_loglin(
    outcome_space,
    encoder,
    decision_rule,
    model_family,
    seeds: list,
    num_folds: int,
    results_folder,
):
    """
    Ensemble via MajorityVoteClassifier a number of instances of logistic regressors, all stored in the results folder.

    outcome_space: NamedClasses for the dataset
    encoder: text encoder used in training
    decision_rule: how to map 1 pmf to 1 label
    model_family: the string you used to name a subfolder under the results folder
    seeds: a list of seeds (they name subfolders under the model_family folder)
    num_folds: number of folds in cross validation (these name subfolders under the seed folder)
    results_folder: where everything is saved
    """
    models = []
    for seed in seeds:
        for f in range(num_folds):
            models.append(
                WrapperForLoglinearModel(
                    encoder,
                    loglin.load(
                        f"{results_folder}/{model_family}/seed-{seed}/fold-{f}/model"
                    ),
                )
            )
    return clslib.MajorityVoteClassifier(outcome_space, models, decision_rule)


class WrapperForNeuralModel(clslib.ProbabilisticModel):
    """
    Wrapper to treat an instance of our torch softmax regressor in the API of a ProbabilisticModel
    """

    def __init__(self, encoder, model):
        """text encoder used for training and trained model"""
        self.encoder = encoder
        self.model = model

    def __call__(self, x):
        """Predicts a pmf for x"""
        r = smlp.predict_probs(
            [x], self.model, self.encoder, batch_size=1
        )  # of course, batching multiple texts is faster for a large dataset, but this is more didactic
        return r[0]


def ensemble_neural(
    outcome_space,
    encoder,
    decision_rule,
    model_family,
    seeds: list,
    num_folds: int,
    results_folder,
):
    """
    Ensemble via MajorityVoteClassifier a number of instances of neural softmax regressors, all stored in the results folder.

    outcome_space: NamedClasses for the dataset
    encoder: text encoder used in training
    decision_rule: how to map 1 pmf to 1 label
    model_family: the string you used to name a subfolder under the results folder
    seeds: a list of seeds (they name subfolders under the model_family folder)
    num_folds: number of folds in cross validation (these name subfolders under the seed folder)
    results_folder: where everything is saved
    """
    models = []
    for seed in seeds:
        for f in range(num_folds):
            models.append(
                WrapperForNeuralModel(
                    encoder,
                    smlp.load(
                        f"{results_folder}/{model_family}/seed-{seed}/fold-{f}/model"
                    ),
                )
            )
    return clslib.MajorityVoteClassifier(outcome_space, models, decision_rule)
