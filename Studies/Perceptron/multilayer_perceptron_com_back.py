import numpy as np
import pandas as pd
from typing import Tuple
from copy import deepcopy

class Perceptron:
    def __init__(
        self, n_layers_: int = 1, n_epochs_: int = 2, n_outputs_: int = 1,
        eta_: int = 0.1, bias_: float = 1.0, w = None
    ) -> None:
        self.weights = np.array([[[np.random.random() for _ in range(n_epochs_)]
                                for _ in range(n_epochs_)]
                                for _ in range(n_layers_)])
        if w:
            if len(w) != n_epochs_:
                raise ValueError(f'Length doesnt match number of neurons: n_epochs = {n_epochs_} | shape={len(w)}')
            self.weights[0][0] = np.array(w)
        self.n_layers = n_layers_
        self.n_epochs = n_epochs_
        self.neuron = np.ndarray((n_layers_, n_epochs_), dtype=np.float64)
        self.n_outputs = n_outputs_
        self.outputs = np.ndarray((n_outputs_, ), dtype=np.float64)
        self.eta = eta_
        self.bias = np.array([[bias_ for _ in range(n_epochs_)] for _ in range(n_layers_)])

    def f(self, x: float) -> int:
        return 1 if x > 0.5 else 0

    def sigmoid(self, x: float) -> float:
        return 1/(1 + np.exp(-x))

    def dsigmoid(self, x: float) -> float:
        return x * (1 - x)

    def feed_forward(self, inputs: np.array) -> None:
        if inputs.shape[0] != self.n_epochs:
            raise ValueError(f'Shape doesnt match number of neurons: n_epochs = {self.n_epochs} | shape={inputs.shape}')
        self.neuron[0] = deepcopy(inputs)
        for l in range(self.n_layers):
            for e in range(self.n_epochs):
                net = 0.0
                for j in range(self.n_epochs):
                    net += self.neuron[l][j] * self.weights[l][e][j]
                net += self.bias[l][e]
                Y = self.sigmoid(net)
                if l == self.n_layers - 1:
                    for i in range(self.n_outputs):
                        self.outputs[i] = Y
                else:
                    self.neuron[l+1][e] = Y

    def backpropagation(self, expected: np.array) -> None:
        layer_errors = np.ndarray((self.n_layers+1, self.n_epochs+1))
        for l in range(self.n_layers-1, -1, -1):
            delta = []
            if l != self.n_layers - 1:
                for e in range(self.n_epochs):
                    gama = 0.0
                    for k in range(self.n_epochs):
                        gama += self.weights[l+1][e][k] * layer_errors[l+1][k]
                    delta.append(gama)
                for e in range(self.n_epochs):
                    layer_errors[l][e] = delta[e] * self.dsigmoid(self.neuron[l+1][e])
            else:
                for i in range(self.n_outputs):
                    delta.append(self.outputs[i] - expected[i])
                for e in range(self.n_epochs):
                    layer_errors[l][e] = delta[e % self.n_outputs] * self.dsigmoid(self.outputs[e % self.n_outputs])
        return layer_errors

    def update(self, delta: np.array) -> None:
        for l in range(self.n_layers):
            for e in range(self.n_epochs):
                for j in range(self.n_epochs):
                    self.weights[l][e][j] -= self.eta * delta[l][e] * self.neuron[l][j]
                self.bias[l][e] -= self.eta * delta[l][e]

    def calc_accuracy(self, d: np.array, predictions: list):
        ok = 0.0
        for i in range(d.shape[0]):
            if d[i] - self.f(predictions[i]) == 0:
                ok += 1
        return ok / d.shape[0]

    def fit(
        self, df: pd.DataFrame, num_iter: int = 10, verbose: bool = False
    ) -> None:
        n, m = df.shape
        cols = df.columns

        inputs = np.ndarray((n, m-1), dtype=np.float64)
        for c in range(m-1):
            inputs[:,c] = df[cols[c]]

        d = np.array(df[cols[-1]], dtype=np.float64)
        print(f'\ninputs:\n{inputs}\nd:\n{d}\nweights:\n{self.weights}\n')
        
        for k in range(num_iter):
            Y = []
            for i in range(n):
                self.feed_forward(inputs=inputs[i])
                if not d[i].shape:
                    delta = self.backpropagation(expected=np.array([d[i]]))
                else:
                    delta = self.backpropagation(expected=d[i])
                self.update(delta=delta)
                Y.append(self.outputs[0])
            Y = np.array(Y)
            Accuracy = self.calc_accuracy(d=d, predictions=Y)
            if verbose:
                print(f'Iteração: {k:3}..........Accuracy: {Accuracy * 100}%')
            if Accuracy == 1.0:
                break
        
        print(f'\nResults:\n{np.where(Y > 0.5, 1, 0)}\nd:\n{d}\nweights:\n{self.weights}\n')

def main():
    w = [0.5, 0.5]
    df = pd.read_csv('data_2.csv')
    print(df.head())
    model = Perceptron(n_layers_=1, n_epochs_=2, bias_=0.5, eta_=0.1)
    model.fit(df=df, num_iter=100000, verbose=True)

if __name__ == "__main__":
    main()