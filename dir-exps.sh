# run sgm on undirected graph
# python proc_v_sgm.py --src-embs embs/wiki.en.vec --trg-embs embs/wiki.de.vec --norm unit center unit --function sgm --pairs dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 4000 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1

# run sgm on directed graph
thresholds=(0 1e-5 1e-4 1e-3 1e-2 1e-1) 
for thresh in "${thresholds[@]}"
do
    python dir_sgm.py --norm unit center unit --src-embs embs/wiki.en.vec --trg-embs embs/wiki.de.vec --src en --trg de --pairs dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 4000 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1 --thresh $thresh >> out2.txt
done
