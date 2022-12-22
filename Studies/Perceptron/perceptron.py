#Bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

#Funcao de transferencia
def f_degrau(x: float) -> int:
    return int(x >= 0)

#Funcao para rodar a iteração atual da rede neural
def run_iter(inputs: np.ndarray, weights: np.ndarray,
                  n: float, d: np.array) -> list:
    cols = inputs.shape[0]
    ok = 0
    results = []
    for i in range(cols):
        #Somar os inputs com pesos
        net = inputs[i] * weights
        Y = f_degrau(net.sum())
        #Calcular o erro
        gama = d[i] - Y
        results.append(Y)
        #Se não houve erro
        #adiciona para o calculo da acurácia
        if not gama:
            ok += 1
        #Atulizar os pesos
        weights = weights + n * gama * inputs[i]
    return weights, ok / cols, results

def percepton(df: pd.DataFrame, weights: np.array, num_iter: int, taxa: float, bias: int = 1) -> Tuple[np.array, list]:
    n, m = df.shape
    cols = df.columns
    
    #Criar os inputs
    inputs = np.ndarray((n, m), dtype=np.float32)
    for c in range(m-1):
        inputs[:,c] = df[cols[c]]
    #Adicionar o bias na função
    inputs[::,-1] = bias

    #Valores previstos
    d = np.array(df[cols[-1]], dtype=int)
    print(f'\ninputs:\n{inputs}\nd:\n{d}\nweights:\n{weights}\n')
    
    #Rodar o modelo para o número de iterações ou ao chegar o máximo
    for k in range(num_iter):
        weights, accuracy, Y = run_iter(inputs=inputs, weights=weights, n=taxa, d=d)   
        print(f'Iteração: {k:2} | Accuracy: {accuracy*100}%')
        if accuracy == 1.0:
            break
    return weights, Y

def main():
    str = input('Qual dataset? ')
    df = pd.read_csv(str + '.csv')
    print(df.head())
    w = list(map(float, input('Pesos: ').split()))
    weights = np.array(w, dtype=np.float32)
    n = float(input('Taxa de aprendizagem: '))
    model = percepton(df=df, weights=weights, num_iter=10, taxa=n)
    print(f'Weights: {model[0]}\nResults: {model[1]}')

if __name__ == "__main__":
    main()