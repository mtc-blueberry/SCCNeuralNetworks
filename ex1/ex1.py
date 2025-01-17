import os
import numpy as np
from sklearn.model_selection import train_test_split

def getFileInput(): # essa funcao faz o parseamento dos arquivos de entrada e saida para uso durante execucao
    arr_files  = os.listdir()
    arr_files.sort()
    inputs_per_file = []
    outputs_per_file = []

    for in_file in arr_files: # arquivos de entrada
        if in_file.endswith(".in"):
            with open(in_file) as f:
                read_data = f.read()
                parsed_data = read_data.rstrip().rsplit()
                parsed_data = list(map(int, parsed_data))
                inputs_per_file.append(parsed_data)
        else:
            if in_file.endswith(".out"): # arquivos de saida
                with open(in_file) as f:
                    read_data = f.read()
                    parsed_data = read_data.rstrip().rsplit()
                    parsed_data = list(map(int, parsed_data))
                    outputs_per_file.append(parsed_data)

    inputs_per_file = np.array(inputs_per_file)
    outputs_per_file = np.array(outputs_per_file)
    return inputs_per_file, outputs_per_file

def Adaline(x_in, t, learning_rate):
    #divisao em conjuntos de treinamento e teste de maneira estratificada
    X_train, X_test, t_train, t_test = train_test_split(x_inputs, t, stratify=t)
    weights = np.zeros(len(x_in[0]))
    # training
    weights = LMS(X_train, t_train, learning_rate, weights)
    # test
    test(X_test, t_test, weights)

    return weights

def activation(y): # funcao de ativacao
    if y > 0:
        return 1
    else:
        return -1

def test(x, t, weights): # apenas testa a acurácia do modelo treinado em casos não vistos
    total = len(x)
    correct = 0
    incorrect = 0
    print('Beginning TESTING...')
    for i in range(len(x)): # para todas os samples
        # multiplica cada valor do sample por seu respectivo peso e depois soma os resultados
        s_out = np.sum(np.multiply(x[i], weights))
        s_out = activation(s_out) # passa o resultado para a funcao de ativacao
        if s_out != t[i]:
            incorrect += 1
            print(f'Inputs {i}: Incorrect')
        else:
            correct += 1
            print(f'Inputs {i}: Correct')
    print(f'Number of correct responses: {correct}')
    print(f'Number of INCORRECT responses: {incorrect}')
    accuracy = correct/total
    print(f'Accuracy: {accuracy}')

    return accuracy

# função para o algoritmo LMS
def LMS(x_in, y, learning_rate, weights):
    epochs = 0
    all_correct = False
    print('Beginning TRAINING...')
    while not all_correct: # o loop continua até que se acerte o resultado para todos os inputs
        epochs += 1
        all_correct = True
        for i in range(len(x_in)): # para todos os inputs
            # multiplica cada valor do sample por seu respectivo peso e depois soma os resultados
            s_out = np.sum(np.multiply(x_in[i], weights))
            s_out = activation(s_out) # passa o resultado para a funcao de ativacao
            if s_out != y[i]: # se existe erro, os pesos sao recalculados
                all_correct = False
                error = y[i][0] - s_out
                for u in range(len(weights)):
                    weights[u] = weights[u] + (learning_rate * (y[i][0] - s_out) * x_in[i][u])
                print(f'Epoch {epochs}, inputs {i}: Incorrect')
                print(f'New weights: {weights}')
            else: # se nao existe erro, a saida é igual a esperada
                print(f'Epoch {epochs}, inputs {i}: Correct')

    return weights

learning_rate = 0.5
x_inputs = []
y = []
x_inputs, y = getFileInput()
final_weights = Adaline(x_inputs, y, learning_rate)
print(f'Final weights: {final_weights}')
