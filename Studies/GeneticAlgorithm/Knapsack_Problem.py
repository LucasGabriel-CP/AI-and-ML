#bibliotecas necessárias
import numpy as np
from random import randint
from copy import deepcopy
from functools import partial
from random import sample, choices, random
from typing import List, Tuple, Callable
import time

#Objetos auxiliares
Universo = np.ndarray
Individuo = np.ndarray
Populacao = List[Individuo]
FitnessFunc = Callable[[Individuo], int]
PopulationFunc = Callable[[], Populacao]
SelectFunc = Callable[[Populacao, Universo, FitnessFunc], Tuple[Individuo, Individuo]]
CrossOverFunc = Callable[[Individuo, Individuo], Tuple[Individuo, Individuo]]
MutationFunc = Callable[[Individuo], Individuo]
ValidFunc = Callable[[Individuo, Universo, float], bool]

#Função para criar um individuo
#Crio um individuo preenchendo com os valores de 0 e 1 aleatórios
def GenerateInd(universo: Universo, capacidade: float) -> Individuo:
    tam = universo.shape[0]
    ind = np.zeros(tam)
    while True:
        id = randint(0, tam-1)
        if capacidade < universo[id][1]:
            break
        ind[id] = 1
        capacidade -= universo[id][1]
    return ind

#Função para gerar a primeira população de indivíduos
#Para cada posição, vou criar um novo indivíduo
def GenPop(universo: Universo, capacidade: float, tam: int) -> Populacao:
    return [GenerateInd(universo=universo, capacidade=capacidade) for _ in range(tam)]

#Função para validar se o individuo atente as restrições
def ValidIndividuo(individuo: Individuo, itens: Universo, capacidade: float) -> bool:
    if individuo.shape[0] != itens.shape[0]:
        raise ValueError(f'Tamanhos diferentes. Individuo: {individuo.shape[0]}, universo: {itens.shape[0]}')
    return sum(individuo * itens[:,1]) <= capacidade

#Função que define o "score" de um individuo
#Multiplica o array do individuo pelo preço dos itens
#Soma os valores do array resultante e retorna o resultado
def Fitness(individuo: Individuo, itens: Universo) -> int:
    if individuo.shape[0] != itens.shape[0]:
        raise ValueError(f'Tamanhos diferentes. Individuo: {individuo.shape[0]}, universo: {itens.shape[0]}')
    return sum(individuo * itens[:,0])

#Função de seleção dos pais
#Faço uma escolha random com pesos
def SelectNew(population: Populacao, worst: float, itens: Universo, fitness_func: FitnessFunc) -> Populacao:
    return choices(
        population=population,
        weights=[(fitness_func(individuo=individuo) - worst + 1) for individuo in population],
        k=2
    )

#Função de CrossOver dos pais
#Pega uma porção aleatória entre os pais e troca ela
def CrossOver(a: Individuo, b: Individuo) -> Tuple[Individuo, Individuo]:
    if a.shape[0] != b.shape[0]:
        raise ValueError('Tamanhos diferentes')
    
    #Escolher duas posições aleatórias dos indivíduos
    N = a.shape[0]
    p1, p2 = randint(1, N-1), randint(1, N-1)
    while p1 == p2: p2 = randint(1, N-1)
    if p1 > p2: p1, p2 = p2, p1

    #Crio um novo individuo
    #NewInd1 = b[i], b[i+1], b[i+2], ..., b[p1-1]
    NewInd1 = b[:p1]
    #NewInd1.append(a[p1], a[p1+1], a[p1+2], ..., a[p2-1])
    NewInd1 = np.append(NewInd1, a[p1:p2])
    #NewInd1.append(b[p2], a[p2+1], b[p2+2], ..., b[N-1])
    NewInd1 = np.append(NewInd1, b[p2:])

    #Repete o mesmo processo para NewInd2    
    NewInd2 = a[:p1]
    NewInd2 = np.append(NewInd2, b[p1:p2])
    NewInd2 = np.append(NewInd2, a[p2:])

    if NewInd1.shape[0] != N or NewInd2.shape[0] != N:
        raise ValueError('CrossOver deu ruim')

    return NewInd1, NewInd2

#Função para mutar o indivíduo
#Pega um segmento do individuo e inverte ele
def Mutation(individuo: Individuo) -> Individuo:
    #Escolhe os dois pontos do segmento
    N = individuo.shape[0]
    p1, p2 = randint(1, N-1), randint(1, N)
    while p1 == p2: p2 = randint(1, N)
    if p1 > p2: p1, p2 = p2, p1

    #ind_mutado = individuo[i, i+1, i+2, ..., p1-1]
    ind_mutado = individuo[:p1]
    #ind_mutado.append(individuo[p2-1, p2-2, p2-3, ..., p1])
    ind_mutado = np.append(ind_mutado, individuo[p1:p2][::-1])
    #ind_mutado.append(individuo[p2, p2+1, p2+2, ..., N-1])
    ind_mutado = np.append(ind_mutado, individuo[p2:])
    
    if ind_mutado.shape[0] != N:
        raise ValueError('Mutação deu ruim')
    
    return ind_mutado

#Função para rodar o algoritmo genético
def run_evo(
    populate_func: PopulationFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    mochila: Universo,
    selection_func: SelectFunc = SelectNew,
    crossover_func: CrossOverFunc = CrossOver,
    mutation_func: MutationFunc = Mutation,
    valid_func: ValidFunc = ValidIndividuo,
    generation_limit: int = 100,
    show_progress: bool = False,
    croosover_point: int = 0.65,
    mutation_point: int = 0.4
) -> Tuple[Populacao, int]:
    #Criar a população inicial
    population = populate_func()
    start, stop, max_gen = 0, 0, 0

    #Rodar as gerações
    #Decobrir os melhores indivíduos ordenando usando a função fitness
    population = sorted(
        population,
        key=lambda individuo: fitness_func(individuo=individuo),
        reverse=True
    )
    for i in range(generation_limit):

        
        aux = sorted(
                population,
                key=lambda individuo: fitness_func(individuo=individuo),
                reverse=True
            )
        if show_progress:
            print(f'Generation: {i}. Result: {fitness_func(individuo=aux[0])}. Time: {stop-start}')
        #Se o resultado atingiu o melhor resultado, encerra o loop
        if fitness_func(individuo=aux[0]) >= fitness_limit:
            break

        max_gen += 1
        start = time.time()

        #Guarda os dois melhores resultados
        next_gen = aux[:2]

        worst = fitness_func(aux[-1])

        #Realiza as mutações/crossovers necessários
        for _ in range(int(len(population)/2)-1):
            #Selecionar os pais para o crossover
            parents = selection_func(population=population, worst=worst, itens=mochila, fitness_func=fitness_func)

            #verificar se ta no ponto de crossover
            if random() > croosover_point:
                #Realizar o crossover
                sobrevivente_a, sobrevivente_b = crossover_func(a=parents[0], b=parents[1])
                sobrevivente_a_mutado = np.ones(parents[0].shape)
                sobrevivente_b_mutado = np.ones(parents[0].shape)

                #verificar se ta no ponto de mutação
                if random() > mutation_point:
                    sobrevivente_a_mutado = mutation_func(individuo=sobrevivente_a)
                    sobrevivente_b_mutado = mutation_func(individuo=sobrevivente_b)
                
                # Verificar se o individuo mutado ou cruzado ou o original que deve entrar
                #na próxima geração
                if valid_func(sobrevivente_a_mutado):
                    next_gen.append(sobrevivente_a_mutado)
                elif valid_func(sobrevivente_a):
                    next_gen.append(sobrevivente_a)
                else:
                    next_gen.append(parents[0])
                
                # Verificar se o outro individuo mutado ou cruzado ou o original que deve entrar
                #na próxima geração
                if valid_func(sobrevivente_b_mutado):
                    next_gen.append(sobrevivente_b_mutado)
                elif valid_func(sobrevivente_b):
                    next_gen.append(sobrevivente_b)
                else:
                    next_gen.append(parents[1])
            else:
                sobrevivente_a = np.ones(parents[0].shape)
                sobrevivente_b = np.ones(parents[0].shape)

                #verificar se ta no ponto de mutação
                if random() > mutation_point:
                    sobrevivente_b = mutation_func(individuo=parents[0])
                    sobrevivente_a = mutation_func(individuo=parents[1])
                # Verificar se o outro individuo mutado ou o original que deve entrar
                #na próxima geração
                if valid_func(sobrevivente_a):
                    next_gen.append(sobrevivente_a)
                else:
                    next_gen.append(parents[0])
                
                # Verificar se o outro individuo mutado ou o original que deve entrar
                #na próxima geração
                if valid_func(sobrevivente_b):
                    next_gen.append(sobrevivente_b)
                else:
                    next_gen.append(parents[1])
        
        #Atualizar a população para a próxima geração
        population = deepcopy(next_gen)
        stop = time.time()

    return population, max_gen

def main():
    test_cases = 10
    for tc in range(1, test_cases+1):
        #Carregar os dados na memória
        prices, weights, sol = np.ndarray((0)), np.ndarray((0)), np.ndarray((0))
        file = open(f'./data/p0{tc}_p.txt')
        for line in file:
            if line == '\n':
                break
            prices = np.append(prices, int(line.split()[0]))
        
        file = open(f'./data/p0{tc}_w.txt')
        for line in file:
            if line == '\n':
                break
            weights = np.append(weights, int(line.split()[0]))

        file = open(f'./data/p0{tc}_s.txt')
        for line in file:
            if line == '\n':
                break
            sol = np.append(sol, int(line.split()[0]))
        
        file = open(f'./data/p0{tc}_c.txt')
        for line in file:
            if line == '\n':
                break
            capacidade = int(line.split()[0])

        mochila = np.ndarray((len(prices), 2))
        for i in range(len(prices)):
            mochila[i][0] = prices[i]
        for i in range(len(weights)):
            mochila[i][1] = weights[i]
        
        fitness_limit = Fitness(sol, mochila)
        print(f'Mochila:\nValor x Peso\n{mochila}\nCapacidade: {capacidade}')
        
        #Rodar o GA
        population, max_gen = run_evo(
            populate_func=partial(
                GenPop, universo=mochila, capacidade=capacidade, tam=128
            ),
            fitness_func=partial(
                Fitness, itens=mochila
            ),
            mutation_func=Mutation,
            valid_func=partial(
                ValidIndividuo, itens=mochila, capacidade=capacidade
            ),
            fitness_limit=fitness_limit,
            mochila=mochila,
            generation_limit=10000,
            show_progress=False
        )
        resp = sorted(
                        population,
                        key=lambda individuo: Fitness(individuo, mochila),
                        reverse=True
                    )[0]
        gap = 100.0 - Fitness(resp, mochila)*100/fitness_limit
        print(f'Caso de teste: {tc}\nResultado: {Fitness(resp, mochila)}\nEscolhas: {resp}\nQuantidade de gerações: {max_gen}\nGap: {gap}%')
        print('-'*49)

if __name__ == "__main__":
    main()