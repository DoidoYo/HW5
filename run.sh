clear

source venv/bin/activate

#python3 main.py --knn-k 5 --test-split dev --predictions-file simple-knn-preds.txt
#python3 accuracy.py --labeled data/labels-mnist-dev.npy --predicted simple-knn-preds.txt


#python3 main.py --knn-k 5 --dr-algorithm pca --target-dim 300 --test-split dev --predictions-file pca-knn-preds.txt
#python3 accuracy.py --labeled data/labels-mnist-dev.npy --predicted pca-knn-preds.txt

python3 main.py --knn-k 5 --dr-algorithm lle --lle-k 10 --target-dim 300 --test-split dev \
--predictions-file pca-knn-preds.txt
