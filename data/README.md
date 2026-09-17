# MNIST data

From the repository root, run once:

```sh
bash scripts/download-mnist.sh
```

The script downloads and decompresses four original MNIST files into this
directory. It verifies SHA-256 checksums, skips verified existing files and
refuses to overwrite invalid files or symbolic links. Requires Bash, curl,
gzip and either shasum or sha256sum, available on typical macOS/Linux setups
or in WSL. Downloads require internet access; later runs of the model do not.

```text
data/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

These are ordinary files, not links to another folder. Run the program from
the repository root so its data/ paths resolve correctly. We ignore the
downloaded files in Git; viewers run the same command after cloning.

## Source and attribution

MNIST was created by Yann LeCun, Corinna Cortes and Christopher J.C. Burges.
The original [dataset page](http://yann.lecun.com/exdb/mnist/) describes the
60,000 training and 10,000 test images. We use the CVDF mirror also listed in
the [TensorFlow Datasets MNIST loader](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py).
Decompression does not alter the IDX contents; normalization occurs in C++.

## Redistribution decision

Checked September 15, 2026. License information is inconsistent:
[Hugging Face's MNIST card](https://huggingface.co/datasets/ylecun/mnist)
labels its version MIT, while
[Chalmers C3SE](https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/#mnist)
describes MNIST as CC BY-SA 3.0. We could not retrieve the original author's
page to verify the applicable grant for these original IDX files.

This does not establish that redistribution is prohibited. We have chosen
not to redistribute the dataset in this repository until its terms are
resolved. The repository supplies the downloader and attribution, not a new
license for MNIST. A loader's software license is not a dataset license.
