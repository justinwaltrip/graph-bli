# run sgm on undirected graph
python proc_v_sgm.py --src-embs /home/ubuntu/graph-bli/embs/wiki.en.vec --trg-embs /home/ubuntu/graph-bli/embs/wiki.de.vec --norm unit center unit --function sgm --pairs /home/ubuntu/graph-bli/dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 100 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1

# run sgm on directed graph
python dir_sgm.py --src en --trg de --pairs /home/ubuntu/graph-bli/dicts/en-de/train/en-de.0-5000.txt.1to1 --n-seeds 100 --max-embs 200000 --min-prob 0.0 --proc-iters 1 --softsgm-iters 1 --diff-seeds-for-rev --iterative-softsgm-iters 1 --new-nseeds-per-round -1
