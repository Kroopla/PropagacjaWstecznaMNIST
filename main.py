import numpy as np
import os
from scipy.ndimage import zoom


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.learning_rate = learning_rate
        # Inicjalizacja wag
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        # print(f"Initializing weights 1: {self.weights1}")
        # print(f"Initializing weights 2: {self.weights2}")

    def forward(self, X):
        # print(f'Forward: {X}')
        self.z2 = np.dot(X, self.weights1)  # Wejście do pierwszej wasrtwy ukrytej
        # print(f"z2 shape: {self.z2}")
        self.a2 = self.relu(self.z2)  # Aktywacja pierwzszej warstwu ukyrtej
        # print(f"a2 shape: {self.a2}")
        self.z3 = np.dot(self.a2, self.weights2)  # Wejście do wyjsciowej wasrtwy ukrytej
        # print(f"z3 {self.z3}")
        # print(f"Output before softmax: {self.z3}")
        output = self.softmax(self.z3)  # Aktywacja wyjsciowej warstwu ukyrtej
        # print(f"Output after softmax: {output}")
        return output

    def backward(self, X, y_true, output):
        # WSTECZNA PROPAGACJA!!!

        # 1. Oblicz błędu na wyjściu
        error = output - y_true

        # 2. Pochodna funkcji softmax
        d_softmax = error / output.shape[0]  # Normalizacja przez liczbę przykładów

        # 3. Pochodna funkcji aktywacji ReLU
        d_relu = np.where(self.a2 <= 0, 0, 1)

        # 4. Obliczanie gradientu dla każdej warstwy
        # print("Kształt self.a2:", self.a2.shape)
        # print("Kształt d_softmax:", d_softmax.shape)
        gradient_weights2 = np.dot(self.a2.reshape(1, -1).T, d_softmax.reshape(1, -1))
        X_matrix = X.reshape(-1, 1)
        gradient_weights1 = np.dot(X_matrix.T, np.dot(d_softmax.reshape(1, -1), self.weights2.T) * d_relu)
        # print(f"gradient_weights1: {gradient_weights1}")
        # print(f"gradient_weights2: {gradient_weights2}")

        # 5. Aktualizacja wag
        self.weights1 -= self.learning_rate * gradient_weights1
        self.weights2 -= self.learning_rate * gradient_weights2

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        if x.ndim == 1:  # Jeśli x jest wektorem (1D)
            exp_x = np.exp(x - np.max(x))        # funkcja eksponencjalna(x - największa wartość)
            return exp_x / np.sum(exp_x)         # exp_x / suma eksponencjalnych wartości wzdłóż osi 1
        else:  # Jeśli x jest macierzą (2D)
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # funkcja eksponencjalna(x - największa wartość)
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)   # exp_x / suma eksponencjalnych wartości wzdłóż osi 1

    def cross_entropy(self, y_pred, y_true):
        # print(f"y_pred: {y_pred}")
        # print(f"y_true: {y_true}")
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-15)) / m
        return loss

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            for X, y in zip(X_train, y_train):
                # Propagacja w przód
                output = self.forward(X)

                # Obliczanie straty
                loss = self.cross_entropy(output, y)

                # Wsteczna propagacja i aktualizacja wag
                self.backward(X, y, output)
            if epoch % 100 == 0:
                print(f'Epoka {epoch + 1}/{epochs}, Strata: {loss}')


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


def resize_images(images, new_size):
    resized_images = np.zeros((images.shape[0], new_size, new_size))
    for i in range(images.shape[0]):
        resized_images[i] = zoom(images[i].reshape(28, 28), (new_size / 28, new_size / 28))

    return resized_images.reshape(images.shape[0], new_size * new_size)


# Załadowanie danych i etykiet z plików
train_images, train_labels = load_mnist('.', kind='train')
test_images, test_labels = load_mnist('.', kind='t10k')

# Losowe próbkowanie danych do treningu i testowania
train_indices = np.random.choice(len(train_images), 1000, replace=False)
test_indices = np.random.choice(len(test_images), 50, replace=False)

train_images_sampled = train_images[train_indices] / 255.0
train_labels_sampled = train_labels[train_indices]
test_images_sampled = test_images[test_indices] / 255.0
test_labels_sampled = test_labels[test_indices]

new_size = 10  # Nowy rozmiar obrazów, np. 10x10
train_images_resized = resize_images(train_images_sampled, new_size)
test_images_resized = resize_images(test_images_sampled, new_size)

# print("Rozmiar danych treningowych (obrazy):", train_images.shape)
# print("Rozmiar danych treningowych (etykiety):", train_labels.shape)
# print("Rozmiar danych testowych (obrazy):", test_images.shape)
# print("Rozmiar danych testowych (etykiety):" test_labels.shape)

# One-hot encoding etykiet
train_labels_one_hot = one_hot_encode(train_labels_sampled, 10)
test_labels_one_hot = one_hot_encode(test_labels_sampled, 10)

# Ilość wejść, neuronów w warstwie ukrytej, wyjść, learning rate
neural_network = NeuralNetwork(100, 32, 10, 0.01)

# Trenowanie sieci przez daną ilość epok epok
neural_network.train(train_images_resized, train_labels_one_hot, epochs=1000)

# Iteracja przez obrazy testowe i etykiety, zapisywanie wyników predykcji i wypisanie poprawności

accuracy = 0

for X, y in zip(test_images_resized, test_labels_one_hot):
    prediction = neural_network.forward(X)
    predicted_class = np.argmax(prediction)
    if np.argmax(y) == predicted_class:
        accuracy += 1
    print(f"Prediction:\n {prediction} \n(expected:\n {y})\n predicted_class: {predicted_class}")

print(f"Accuracy: {accuracy}, a to jest {accuracy/50*100}%")