import time
import ray
import numpy as np
import json
import math
from pathlib import Path
import tqdm
import itertools
import tensorflow_datasets as tfds


def process_dict_pairs(pair_file):
    """Parses a dictionary pairs file.

    Pairs as list of (srcwd, trgwd) tuples
    L1 and L2 vocabularies as sets.
    """
    pairs = []
    l1_words = set()
    l2_words = set()
    with open(pair_file) as f:
        for line in f:
            w1, w2 = line.split()
            w1 = w1.strip()
            w2 = w2.strip()
            pairs.append((w1, w2))
            l1_words.add(w1)
            l2_words.add(w2)
    return pairs, l1_words, l2_words


def load_wiki_data(lang):
    """
    Load fastText wiki data for relevant languages
    """
    # TODO remove [:1%]
    # load wiki40b data for language
    ds = tfds.load(f"wiki40b/{lang}", split="train[:1%]")

    # separate samples by special markers
    special_markers = [
        "_START_ARTICLE_",
        "_START_SECTION_",
        "_START_PARAGRAPH_",
        "_NEWLINE_",
    ]

    # TODO add tqdm
    wiki_data = []
    for example in ds:
        text = example["text"].numpy().decode("utf-8")
        text = text.replace("\n", "")
        for marker in special_markers:
            text = text.replace(marker, "[SEP]")
        docs = list(filter(lambda x: x != "", text.split("[SEP]")))
        wiki_data.extend(docs)

    return wiki_data


def compute_unigram_counts(lang, words, wiki_data):
    """
    Compute unigram counts for words in a language
    """
    unigram_counts = {}
    for w in tqdm.tqdm(words):
        # compute monogram count for "w"
        count = 0
        for doc in wiki_data:
            count += doc.count(w)
        unigram_counts[w] = count

    return unigram_counts


@ray.remote
def count_bigrams(bigrams, wiki_data):
    """
    Compute bigram count for words in a language
    """
    bigram_counts = {}
    
    for w1, w2 in bigrams:
        count = 0
        for doc in wiki_data:
            count += doc.count(f"{w1} {w2}")
        bigram_counts[str((w1, w2))] = count
    
    return bigram_counts


def _count_bigram(w1, w2, wiki_data):
    """
    Compute bigram count for words in a language
    """
    count = 0
    for doc in wiki_data:
        count += doc.count(f"{w1} {w2}")
    return count


def compute_bigram_counts(lang, words, wiki_data):
    """
    Compute bigram counts for words in a language
    """
    ray.init()

    bigram_counts = {}

    batch_size = 100
    perms = list(itertools.permutations(words, 2))

    results = []
    for i in range(math.perm(len(words), 2) // batch_size + 1):
        bigrams = perms[i * batch_size : (i + 1) * batch_size]
        results.append(count_bigrams.remote(bigrams, wiki_data))
    
    for result in tqdm.tqdm(results):
        bigram_counts.update(ray.get(result))

    return bigram_counts


def get_unigram_counts(lang, words, wiki_data):
    """
    Load unigram counts for words in a language
    """
    unigram_path = Path(f"unigram_counts/{lang}.json")
    unigram_path.parent.mkdir(parents=True, exist_ok=True)

    if unigram_path.exists():
        print(f"Loading unigram counts for {lang}")
        with open(unigram_path) as f:
            unigram_counts = json.load(f)
    else:
        print(f"Computing unigram counts for {lang}")

        unigram_counts = compute_unigram_counts(lang, words, wiki_data)

        # save unigram counts
        with open(unigram_path, "w") as f:
            json.dump(unigram_counts, f, indent=4)

    return unigram_counts


def get_bigram_counts(lang, words, wiki_data):
    """
    Load bigram counts for words in a language
    """
    bigram_path = Path(f"bigram_counts/{lang}.json")
    bigram_path.parent.mkdir(parents=True, exist_ok=True)

    if bigram_path.exists():
        print(f"Loading bigram counts for {lang}")
        with open(bigram_path) as f:
            bigram_counts = json.load(f)
    else:
        print(f"Computing bigram counts for {lang}")

        bigram_counts = compute_bigram_counts(lang, words, wiki_data)

        # save bigram counts
        with open(bigram_path, "w") as f:
            json.dump(bigram_counts, f, indent=4)

    return bigram_counts


def save_word_indices(lang, words):
    """
    Save word indices for a language
    """
    word_ix_path = Path(f"word_indices/{lang}.json")
    word_ix_path.parent.mkdir(parents=True, exist_ok=True)

    word_ix = {w: i for i, w in enumerate(words)}

    # save word indices
    with open(word_ix_path, "w") as f:
        json.dump(word_ix, f, indent=4)


def main():
    """
    1. Load bilingual dictionaries for relvant language comparisons
    2. Load fastText wiki data for each language
    3. For each word pair (w1, w2) in each language:
        add directed edge e from w1 to w2 where w(e) = p(w2|w1) = p(w1, w2) / p(w1) (and vice versa)
    4. Run SGM on directed adjacency matrices
    5. Evaluate performance
    """
    src = "en"
    trg = "de"

    word_pairs, src_words, trg_words = process_dict_pairs(
        f"dicts/{src}-{trg}/train/{src}-{trg}.0-5000.txt.1to1"
    )

    # TODO remove
    src_words = list(sorted(src_words))[:50]
    trg_words = list(sorted(trg_words))[:50]

    for lang, words in [(src, src_words), (trg, trg_words)]:
        wiki_data = load_wiki_data(lang)

        # TODO remove
        wiki_data = wiki_data[:100000]

        unigram_counts = get_unigram_counts(lang, words, wiki_data)
        bigram_counts = get_bigram_counts(lang, words, wiki_data)

        # save word indices used for adjacency matrix
        save_word_indices(lang, words)

        # create directed adjacency matrix
        adj_matrix_path = Path(f"adj_matrices/{lang}.npy")
        adj_matrix_path.parent.mkdir(parents=True, exist_ok=True)

        if adj_matrix_path.exists():
            print(f"Loading adjacency matrix for {lang}")
            with open(adj_matrix_path) as f:
                adj_matrix = np.load(f)
        else:
            print(f"Computing adjacency matrix for {lang}")

            adj_matrix = np.zeros((len(words), len(words)))
            for w1, w2 in tqdm.tqdm(
                itertools.permutations(words, 2), total=math.perm(len(words), 2)
            ):
                w1_ix = words.index(w1)
                w2_ix = words.index(w2)

                # compute p(w2|w1) and p(w1|w2)
                if unigram_counts[w1] != 0:
                    w2_given_w1 = bigram_counts[str((w1, w2))] / unigram_counts[w1]
                else:
                    w2_given_w1 = 0
                if unigram_counts[w2] != 0:
                    w1_given_w2 = bigram_counts[str((w2, w1))] / unigram_counts[w2]
                else:
                    w1_given_w2 = 0

                # add directed edge from w1 to w2
                adj_matrix[w1_ix][w2_ix] = w2_given_w1

                # add directed edge from w2 to w1
                adj_matrix[w2_ix][w1_ix] = w1_given_w2

            # save adjacency matrix
            with open(adj_matrix_path, "w") as f:
                np.save(f, adj_matrix)


if __name__ == "__main__":
    main()
