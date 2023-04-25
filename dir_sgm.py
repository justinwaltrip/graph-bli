import argparse
import random
import sgm
import ray
import numpy as np
import json
import math
from pathlib import Path
import tqdm
import itertools
import tensorflow_datasets as tfds
from utils import matops
from third_party import combine_bidirectional_alignments


def symmetrize(hyps, hyps_rev, heuristic):
    if heuristic == "grow-diag-final":
        return combine_bidirectional_alignments.grow_diag_final(hyps, hyps_rev)
    elif heuristic == "intersection":
        return hyps.intersection(hyps_rev)


def get_topk_hypotheses_from_probdist(hyp_probdist, k=1, minprob=0.0):
    """Returns topk hyps per src word over a minprob from SoftSGM prob. dist."""
    hyp_probdist_topk_over_minprob = matops.keep_topk_over_minprob(
        hyp_probdist, k, minprob
    )
    nonzero_indices = np.nonzero(hyp_probdist_topk_over_minprob)
    hyp_src_inds = nonzero_indices[0].tolist()
    hyp_trg_inds = nonzero_indices[1].tolist()
    hyps = set(zip(hyp_src_inds, hyp_trg_inds))
    return hyps


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


def compute_unigram_counts(words, wiki_data):
    """
    Compute unigram counts for words in a language
    """
    unigram_counts = {}
    for w in tqdm.tqdm(words):
        # compute monogram count for "w"
        count = 0
        for doc in wiki_data:
            # TODO ensure w is not a substring of another word
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
            # TODO ensure w1 w2 are both not substrings of another word
            count += doc.count(f"{w1} {w2}")
        bigram_counts[str((w1, w2))] = count
    
    return bigram_counts


def compute_bigram_counts(words, wiki_data):
    """
    Compute bigram counts for words in a language
    """
    ray.init(ignore_reinit_error=True)

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

        unigram_counts = compute_unigram_counts(words, wiki_data)

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

        bigram_counts = compute_bigram_counts(words, wiki_data)

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


def get_adj_matrix(lang, words, unigram_counts, bigram_counts):
    """
    Load or compute adjacency matrix for a language
    """
    adj_matrix_path = Path(f"adj_matrices/{lang}.npy")
    adj_matrix_path.parent.mkdir(parents=True, exist_ok=True)

    if adj_matrix_path.exists():
        print(f"Loading adjacency matrix for {lang}")
        with open(adj_matrix_path, "rb") as f:
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
        with open(adj_matrix_path, "wb") as f:
            np.save(f, adj_matrix)
    
    return adj_matrix


def unzip_pairs(pairs):
    """Unzips a set of (x, y) pairs to lists of [x1, ..., xn], [y1, ..., yn]"""
    x_list = list(list(zip(*pairs))[0])
    y_list = list(list(zip(*pairs))[1])
    return x_list, y_list


def get_seeds(
    x_seed_inds,
    y_seed_inds,
    gold_x_seed_inds,
    gold_y_seed_inds,
    max_seeds_to_add,
    i,
    total_i,
    always_use_gold=True,
):
    """Get correct number of seeds for a given round.

    Args:
        x_seed_inds: indices for seeds for x space.
        y_seed_inds: indices for seeds for y space.
        gold_x_seed_inds: gold seed indicies in x space.
        gold_y_seed_inds: gold seed indicies in y space.
        max_seeds_to_add: max # of seeds to add per round. Can be an
            integer if same for all rounds or list if different. -1 == all.
        i: index to pull number of seeds from.
        total_i: total number of rounds algorithm will go through.
        always_use_gold: whether or not to always use gold seeds.

    Returns:
        x_seed_inds: indices for seeds for x space.
        y_seed_inds: indices for seeds for y space.
    """
    print("Choosing Seeds...")
    xy_pairs = list(zip(x_seed_inds, y_seed_inds))
    # Extra make sure these are shuffled, in case python does any weird caching.
    random.shuffle(xy_pairs)
    # Filter out potential x,y pairs where either x or y is in the gold seed
    # indices (we know this cannot be correct, and do not want duplicate
    # src/trg words in our seed set).
    xy_nongold_pairs = list(
        filter(
            lambda pair: pair[0] not in set(gold_x_seed_inds)
            and pair[1] not in set(gold_y_seed_inds),
            xy_pairs,
        )
    )

    gold_pairs = set(zip(gold_x_seed_inds, gold_y_seed_inds))
    pairs = list(gold_pairs) + list(xy_nongold_pairs)
    print("\t# of seeds available:", len(pairs))

    if not always_use_gold:
        random.shuffle(pairs)
    if max_seeds_to_add is None:
        num_seeds_to_add = None  # Return all seeds.
    elif isinstance(max_seeds_to_add, int) and max_seeds_to_add >= 0:
        num_seeds_to_add = max_seeds_to_add
    elif isinstance(max_seeds_to_add, int) and max_seeds_to_add < 0:
        num_seeds_to_add = None  # Return all seeds.
    else:
        if len(max_seeds_to_add) == 1:
            num_seeds_to_add = max_seeds_to_add[0]
        else:
            assert len(max_seeds_to_add) == total_i  # must specify for every round.
            num_seeds_to_add = max_seeds_to_add[i - 1]  # zero-indexed.
        if num_seeds_to_add < 0:
            num_seeds_to_add = None
    pairs = pairs[:num_seeds_to_add]
    print("\t# of seeds chosen for round {0}:".format(i), len(pairs))
    return pairs


def run_softsgm_topk(
    x_sim,
    y_sim,
    x_seed_inds=[],
    y_seed_inds=[],
    iters=1,
    k=1,
    minprob=0.0,
    val_set=None,
):
    """Runs SoftSGM and returns topk hyps over minprob per source word in x_sim.

    Args:
        x_sim: normalized embeddings as a distance (similarity) matrix.
        y_sim: normalized embeddings as a distance (similarity) matrix.
        x_seed_inds: indices for seeds for x_sim.
        y_seed_inds: indices for seeds for y_sim.
        iters: how many iterations of softSGM to run and average.
        k: how many hypotheses to take for each source word from prob. dist
            returned from sgm.softsgm.
        minprob: min probability necessary for hypothesis to be considered.
        val_set: validation set as set of (x1, y1) tuples.
    Returns:
        topk hypotheses over a minimum probability per source word in x, as
            list of (src, trg) tuples.
    """
    print("Running SoftSGM Topk.")
    hyp_probdist, all_hyps = sgm.softsgm(x_sim, y_sim, x_seed_inds, y_seed_inds, iters)
    hyps = get_topk_hypotheses_from_probdist(hyp_probdist, k, minprob)
    if val_set:
        dev_src_inds, _ = unzip_pairs(val_set)
        dev_hyps = [hyp for hyp in hyps if hyp[0] in dev_src_inds]
        matches, precision, recall = eval(dev_hyps, val_set)
        print("\tPrecision: {0}%  Recall {1}%".format(precision, recall))
    return hyps


def eval_symm(val_set, hyps, hyps_rev, hyps_int, hyps_gdf=None):
    """Evaluates forward, reverse, and joint hypotheses given a validation set.

    Args:
        hyps: forward hypotheses.
        hyps_rev: reverse hypotheses.
        hyps_int: joint hypotheses (intersection of fwd & rev).
        hyps_gdf: fwd & rev hypotheses symmetrized with grow-diag-final.
        val_set: validation set as set of (x1, y1) tuples.

    Returns:
        Prints precision & recall for all sets of hypotheses.
        Returns (True matches, precision, recall) tuple for each set.
    """
    print("\nRunning Evaluation....")
    dev_src_inds, dev_trg_inds = unzip_pairs(val_set)

    print("\nForward:")
    dev_hyps = set(hyp for hyp in hyps if hyp[0] in dev_src_inds)
    matches, prec, recall = eval(dev_hyps, val_set)
    print(
        "\tPairs matched: {0} \n\t(Precision; {1}%) (Recall: {2}%)".format(
            len(matches), prec, recall
        ),
        flush=True,
    )

    print("\nReverse:")
    dev_hyps_rev = set(hyp for hyp in hyps_rev if hyp[1] in dev_trg_inds)
    matches_rev, prec_rev, recall_rev = eval(dev_hyps_rev, val_set)
    print(
        "\tPairs matched: {0} \n\t(Precision; {1}%) (Recall: {2}%)".format(
            len(matches_rev), prec_rev, recall_rev
        ),
        flush=True,
    )

    print("\nIntersection:")
    dev_hyps_int = dev_hyps.intersection(dev_hyps_rev)
    matches_int, prec_int, recall_int = eval(dev_hyps_int, val_set)
    print(
        "\tPairs matched: {0} \n\t(Precision; {1}%) (Recall: {2}%)".format(
            len(matches_int), prec_int, recall_int
        ),
        flush=True,
    )

    return (
        (matches, prec, recall),
        (matches_rev, prec_rev, recall_rev),
        (matches_int, prec_int, recall_int),
    )


def eval(hypotheses, test_set):
    # Given hypotheses and a test set, returns the matches and percentage
    # matched.
    print(
        "\tLength of hypotheses: {0}\n\tLength of test set: "
        "{1}".format(len(hypotheses), len(test_set)),
        flush=True,
    )
    matches = set(test_set).intersection(hypotheses)
    if hypotheses:
        precision = round((float(len(matches)) / len(hypotheses) * 100), 4)
    else:
        precision = None
    recall = round((float(len(matches)) / len(test_set) * 100), 4)
    return matches, precision, recall


def iterative_softsgm(
    x_sim,
    y_sim,
    input_x_seed_inds=[],
    input_y_seed_inds=[],
    gold_x_seed_inds=[],
    gold_y_seed_inds=[],
    softsgm_iters=10,
    k=1,
    minprob=0.0,
    val_set=None,
    max_seeds_to_add=-1,
    curr_i=1,
    total_i=10,
    diff_seeds_for_rev=False,
    run_reverse=False,
    active_learning=False,
    truth_for_active_learning=None,
):
    """Iteratively runs the SoftSGM (Algorithm 3) from Fishkind et al. (2019),
    feeding in intersection of hypotheses from both directions as seeds for
    the next round. Internally, runs SoftSGM with iters iterations and
    returns top k hypotheses over minprob. Note that setting softsgm_iters=0
    total_i=1 runs vanilla SGM.

    Args:
        x_sim: normalized embeddings as a distance (similarity) matrix.
        y_sim: normalized embeddings as a distance (similarity) matrix.
        input_x_seed_inds: indices for seeds for x_sim.
        input_y_seed_inds: indices for seeds for y_sim.
        gold_x_seed_inds: gold seed indicies in x space.
        gold_y_seed_inds: gold seed indicies in y space.
        softsgm_iters: how many iterations of softSGM to run and average.
        k: how many hypotheses to take for each source word from prob. dist
            returned from sgm.softsgm.
        minprob: min probability necessary for hypothesis to be considered.
        max_seeds_to_add: max # of seeds to add per round. Can be an
            integer if same for all rounds or list if different. -1 == all.
        val_set: validation set as set of (x1, y1) tuples.
        curr_i: current iteration number.
        total_i: total number of iterations that will run.
        run_reverse: If total_i = 1, still runs the reverse direction.
        active_learning: If True, only hypotheses that are correct (either
            in the train or dev set) used as seeds for next iteration.
        truth_for_active_learning: True pairs to be be compared with for
            active learning. If a hypothesis is in this set, use it.

    Returns:
        Hypothesized matches induced in fwd direction for all rows in x.
        Hypothesized matches induced in rev direction for all rows in y.
        Intersection of the above hypothesized matches.
        (all are returned as sets of (x_position, y_position) tuples)
    """
    print("----------------------------------")
    print("\nRound {0} of Iterative SoftSGM\n".format(curr_i))
    print("----------------------------------")
    x_seed_inds, y_seed_inds = unzip_pairs(
        get_seeds(
            input_x_seed_inds,
            input_y_seed_inds,
            gold_x_seed_inds,
            gold_y_seed_inds,
            max_seeds_to_add,
            curr_i,
            total_i,
            True,
        )
    )

    print("Running SoftSGM Forward", flush=True)
    hyps = run_softsgm_topk(
        x_sim, y_sim, x_seed_inds, y_seed_inds, softsgm_iters, k, minprob
    )
    if total_i > 1 or run_reverse:
        print("Running SoftSGM Reverse", flush=True)
        if diff_seeds_for_rev:
            print("Getting different seeds for reverse direction.")
            x_seed_inds, y_seed_inds = unzip_pairs(
                get_seeds(
                    input_x_seed_inds,
                    input_y_seed_inds,
                    gold_x_seed_inds,
                    gold_y_seed_inds,
                    max_seeds_to_add,
                    curr_i,
                    total_i,
                    True,
                )
            )
        hyps_rev = run_softsgm_topk(
            y_sim, x_sim, y_seed_inds, x_seed_inds, softsgm_iters, k, minprob
        )
        hyps_rev = {(i[1], i[0]) for i in hyps_rev}
        hyps_int = symmetrize(hyps, hyps_rev, "intersection")

        if val_set:
            eval_symm(val_set, hyps, hyps_rev, hyps_int)
    else:
        hyps_rev = None
        hyps_int = None
        if val_set:
            print("Evalling Forward")
            dev_src_inds, _ = unzip_pairs(val_set)
            dev_hyps = [hyp for hyp in hyps if hyp[0] in dev_src_inds]
            matches, precision, recall = eval(dev_hyps, val_set)
            print("\tPrecision: {0}%  Recall {1}%".format(precision, recall))

    if curr_i == total_i:
        return hyps, hyps_rev, hyps_int

    curr_i += 1
    if active_learning:
        correct_hyps = set(truth_for_active_learning).intersection(hyps)
        correct_hyps_rev = set(truth_for_active_learning).intersection(hyps_rev)
        joint_x_hyp_pos, joint_y_hyp_pos = unzip_pairs(
            correct_hyps.union(correct_hyps_rev)
        )
    else:
        joint_x_hyp_pos, joint_y_hyp_pos = unzip_pairs(hyps_int)
    return iterative_softsgm(
        x_sim,
        y_sim,
        joint_x_hyp_pos,
        joint_y_hyp_pos,
        gold_x_seed_inds,
        gold_y_seed_inds,
        softsgm_iters,
        k,
        minprob,
        val_set,
        max_seeds_to_add,
        curr_i,
        total_i,
        diff_seeds_for_rev,
        run_reverse,
        active_learning,
        truth_for_active_learning,
    )


def pairs_to_embpos(pairs, src_word2ind, trg_word2ind):
    """Translates a list of (srcwd, trgwd) tuples to a list of
    (src_pos, trg_pos) from the embedding space.

    Args:
        pairs: list of (srcwd, trgwd) tuples
        src_word2ind: source word to index dictionary.
        trg_word2ind: target word to index dictionary.

    Returns:
        List of (src_pos, trg_pos) from the embedding space.

    """
    return list(map(lambda x: (src_word2ind[x[0]], trg_word2ind[x[1]]), pairs))


def create_train_dev_split(pairs, n_seeds, src_word2ind, trg_word2ind, rand=False):
    # Separates input pairs into a train/dev split.
    # Returns tuples of ([src/trg]_train_inds, [src/trg]_dev_inds)
    # If rand=True, randomize seeds picked. Otherwise, choose in order.
    print("Creating the train/dev split given input seeds...")
    if rand:
        random.shuffle(pairs)
    train_pairs = pairs[:n_seeds]
    dev_pairs = pairs[n_seeds:]
    train_inds = pairs_to_embpos(train_pairs, src_word2ind, trg_word2ind)
    dev_inds = pairs_to_embpos(dev_pairs, src_word2ind, trg_word2ind)
    print("Done creating the train/dev split given input seeds.")
    return (train_pairs, dev_pairs), (train_inds, dev_inds)


def main(args):
    """
    1. Load bilingual dictionaries for relvant language comparisons
    2. Load fastText wiki data for each language
    3. For each word pair (w1, w2) in each language:
        add directed edge e from w1 to w2 where w(e) = p(w2|w1) = p(w1, w2) / p(w1) (and vice versa)
    4. Run SGM on directed adjacency matrices
    5. Evaluate performance
    """
    src = args.src
    trg = args.trg

    word_pairs, src_words, trg_words = process_dict_pairs(
        f"dicts/{src}-{trg}/train/{src}-{trg}.0-5000.txt.1to1"
    )

    # TODO remove
    word_pairs = word_pairs[:100]

    # TODO remove
    # get only the words that appear in the word pairs
    src_words = [pair[0] for pair in word_pairs]
    trg_words = [pair[1] for pair in word_pairs]

    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # TODO uncomment
    # _, (train_inds, dev_inds) = create_train_dev_split(
    #     word_pairs, args.n_seeds, src_word2ind, trg_word2ind, args.randomize_seeds
    # )
    _, (train_inds, dev_inds) = create_train_dev_split(
        word_pairs, 10, src_word2ind, trg_word2ind, args.randomize_seeds
    )

    gold_src_train_inds, gold_trg_train_inds = unzip_pairs(train_inds)

    adj_matrices = {}

    for lang, words in [(src, src_words), (trg, trg_words)]:
        wiki_data = load_wiki_data(lang)

        # TODO remove
        wiki_data = wiki_data[:100000]

        unigram_counts = get_unigram_counts(lang, words, wiki_data)
        bigram_counts = get_bigram_counts(lang, words, wiki_data)

        # save word indices used for adjacency matrix
        save_word_indices(lang, words)

        # compute adjacency matrix
        adj_matrices[lang] = get_adj_matrix(lang, words, unigram_counts, bigram_counts)

    # run SGM on adjacency matrices
    hyps, _, _ = iterative_softsgm(
        adj_matrices[src],
        adj_matrices[trg],
        gold_src_train_inds,
        gold_trg_train_inds,
        gold_src_train_inds,
        gold_trg_train_inds,
        args.softsgm_iters,
        args.k,
        args.min_prob,
        dev_inds,
        args.new_nseeds_per_round,
        curr_i=1,
        total_i=args.iterative_softsgm_iters,
        diff_seeds_for_rev=args.diff_seeds_for_rev,
        active_learning=args.active_learning,
        truth_for_active_learning=set(train_inds + dev_inds),
    )

    dev_src_inds, dev_trg_inds = unzip_pairs(dev_inds)
    dev_hyps = set(hyp for hyp in hyps if hyp[0] in dev_src_inds)
    matches, precision, recall = eval(dev_hyps, dev_inds)
    print(
        "\tDev Pairs matched: {0} \n\t(Precision; {1}%) (Recall: {2}%)".format(
            len(matches), precision, recall
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAP Experiments")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source language (e.g. en, de, es, fr, ru, zh)",
    )
    parser.add_argument(
        "--trg",
        type=str,
        required=True,
        help="Target language (e.g. en, de, es, fr, ru, zh)",
    )
    parser.add_argument(
        "--max-embs",
        type=int,
        default=200000,
        help="Maximum num of word embeddings to use.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="The minimum probability to consider for softsgm",
    )
    parser.add_argument(
        "--pairs", metavar="PATH", required=True, help="train seeds + dev pairs"
    )
    parser.add_argument(
        "--n-seeds", type=int, required=True, help="Num train seeds to use"
    )
    parser.add_argument(
        "--proc-iters",
        type=int,
        default=10,
        help="Rounds of iterative Procrustes to run.",
    )
    parser.add_argument(
        "--iterative-softsgm-iters",
        type=int,
        default=1,
        help="Rounds of iterative SoftSGM to run.",
    )
    parser.add_argument(
        "--softsgm-iters",
        type=int,
        default=1,
        help="Rounds of SoftSGM to run to create probdist.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="How many hypotheses to return per source word.",
    )
    parser.add_argument(
        "--randomize-seeds",
        action="store_true",
        help="If set, randomizes the seeds to use (instead of getting them in "
        "order from args.pairs file)",
    )
    parser.add_argument(
        "--new-nseeds-per-round",
        metavar="N",
        type=int,
        nargs="+",
        default=-1,
        help="Number of seeds to add per round in iterative runs.",
    )
    parser.add_argument(
        "--diff-seeds-for-rev",
        action="store_true",
        help="When running matching in reverse, regenerate seeds (if there are "
        "additional input seeds from a previous round, these will then be "
        "shuffled.",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="Whether or not to do active learning",
    )

    args = parser.parse_args()

    main(args)
