for lang in en de ru 
do
	wget \
		-c \
		-P /scratch4/danielk/jwaltri2/embs/ \
		https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.$lang.vec 
done
