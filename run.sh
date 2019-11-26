clear

source venv/bin/activate

python3 main.py --knn-k 5 --test-split data/dev --predictions-file simple-knn-preds.txt

#python3 accuracy.py --labeled data/labels-mnist-dev.npy --predicted simple-knn-preds.txt


#test