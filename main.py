import numpy as np
import os
import gzip


def load_mnist(path, kind='train'):
    """Wczytaj nieskompresowane pliki MNIST."""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


# Przykład użycia
train_images, train_labels = load_mnist('ścieżka_do_folderu', kind='train')
test_images, test_labels = load_mnist('ścieżka_do_folderu', kind='t10k')
