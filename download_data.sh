#!/bin/sh
# Downloads the benchmark files into ./data (UCI Digits ships with scikit-learn).
cd "$(dirname "$0")"; mkdir -p data; cd data
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sfL -o mn_$f https://raw.githubusercontent.com/fgnt/mnist/master/$f
  curl -sfL -o fm_$f https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f
done
for f in elec.csv covtype.csv weather.csv; do
  curl -sfL -o $f https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/$f
done
echo "Data downloaded to ./data"
