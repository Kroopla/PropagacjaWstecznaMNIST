import numpy as np
import os


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicjalizacja wag
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)

    def forward(self, X):
        # Przeprowadź dane przez sieć
        self.z2 = np.dot(X, self.weights1)
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)
        output = self.softmax(self.z3)
        return output

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)


def load_mnist(path, kind='train'):
    """Wczytaj nieskompresowane pliki MNIST."""
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


# Przykład użycia
train_images, train_labels = load_mnist('.', kind='train')
test_images, test_labels = load_mnist('.', kind='t10k')

# print("Rozmiar danych treningowych (obrazy):", train_images.shape)
# print("Rozmiar danych treningowych (etykiety):", train_labels.shape)
# print("Rozmiar danych testowych (obrazy):", test_images.shape)
# print("Rozmiar danych testowych (etykiety):", test_labels.shape)
