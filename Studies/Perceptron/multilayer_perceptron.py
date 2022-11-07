#Modelo MLP para resolver a porta XOR
import numpy as np
import pandas as pd
from typing import Tuple
from copy import deepcopy

#Funcao de transferencia
def f_degrau(x: float) -> int:
    return 1 if x >= 0.0 else 0

#Funcao para calcular a acuracia
def calc_accuracy(d: np.array, predictions: list) -> float:
    ok = 0.0
    for i in range(d.shape[0]):
        if d[i] - predictions[i] == 0:
            ok += 1
    return ok / d.shape[0]

def run_hidden(a: np.array, hidden_weights: np.array, eta: float, d: int) -> list:
    """
        Função para rodar a camada oculta
    Args:
        'hidden_weights' -> pesos dos neurônios
        'a' -> inputs da rede
        'eta' -> taxa de aprendizagem
        'd' -> valor esperado
    returns:
        pesos, resultado
    """

    n_epochs = hidden_weights.shape[0]
    #Calcular o net
    net = 0
    for j in range(n_epochs):
        net += a[j] * hidden_weights[j]
    #Chamar a função de ativacao
    Y = f_degrau(net)

    #Atualizar os pesos
    gama = d - Y
    for j in range(n_epochs):
        hidden_weights[j] += a[j] * gama * eta
    return [hidden_weights, Y]

def run_iter(inputs: np.array, weights: np.array, hidden_weights: np.array, eta: float, d: np.array) -> list:
    """
        Funcao para rodar a iteracao atual
    Args:
        'inputs' -> inputs da rede
        'weights' -> pesos dos neuronios
        'hidden_weights' -> pesos dos neuronios da camada oculta
        'eta' -> taxa de aprendizagem
        'd' -> valor esperado
    returns:
        pesos, acuracia, resultado obtido
    """

    n = inputs.shape[0]
    n_epochs = hidden_weights.shape[0]
    predicted = []
    for i in range(n):
        results, all_weights = [], []
        #Calcular o valor de cada neuronio da proxima camada
        for _ in range(n_epochs):
            #Calcular o net
            net = 0
            for j in range(weights.shape[0]):
                net += inputs[i][j] * weights[j]
            
            #Chamar a funcao de ativacao
            Y = f_degrau(net)
            #Guardar o valor da proxima camada
            results.append(Y)
            
            #Calcular novos pesos
            gama = d[i] - Y
            for j in range(weights.shape[0]):
                weights[j] += inputs[i][j] * gama * eta
            #Guardar o peso
            all_weights.append(deepcopy(weights))
    
        #Rodar a proxima camada
        [hidden_weights, prediction] = run_hidden(a=np.array(results), hidden_weights=hidden_weights, eta=eta, d=d[i])
    
        #Guardar resultado
        predicted.append(prediction)
    
    #Calcular acuracia
    accuracy = calc_accuracy(d=d, predictions=predicted)
    return [all_weights, weights, hidden_weights, accuracy, predicted]

def perceptron(
    df: pd.DataFrame, weights: np.array, num_iter: int,
    eta: float = 0.1, bias: int = 1
) -> Tuple[Tuple[np.array, np.array], np.array]:
    """
        Funcao para criar o modelo
    Args:
        'df' -> dataframe utilizado
        'weights' -> pesos dos neurônios
        'num_iter' -> numero de iteracoes
        'eta' -> taxa de aprendizagem
        'bias' -> bias da rede
    returns:
        pesos, acuracia, resultado obtido
    """
    n, m = df.shape
    cols = df.columns
    
    #Criar os inputs
    inputs = np.ndarray((n, m), dtype=np.float64)
    for c in range(m-1):
        inputs[:,c] = df[cols[c]]
    #Adicionar o bias
    inputs[::,-1] = bias
    
    #Valores previstos
    d = np.array(df[cols[-1]], dtype=np.float64)
    #Criar pesos para a camada oculta
    hidden_weights = np.array([np.random.random() for _ in range(d.shape[0])], dtype=np.float64)
    print(f'\ninputs:\n{inputs}\nd:\n{d}\nweights:\n{weights}\n')
    
    #Rodar o modelo para o número de iterações ou ao chegar o máximo
    for k in range(num_iter):
        all_weights, weights, hidden_weights, accuracy, Y = run_iter(inputs=inputs, weights=weights, hidden_weights=hidden_weights, eta=eta, d=d)
        print(f'Iteração: {k:2} | Accuracy: {accuracy*100}%')
        if accuracy == 1.0:
            break
    Y = np.array(Y)
    return [all_weights, hidden_weights], Y

def main():
    w = [0, 0, -0.5]
    weights = np.array(w, dtype=np.float64)
    df = pd.read_csv('data_3.csv')
    print(df.head())
    model, Y = perceptron(df=df, weights=weights, num_iter=100, eta=0.1)
    print(f'Weights: {model[0]}\nHidden_weights{model[1]}\nResults: {Y}')

if __name__ == "__main__":
    main()