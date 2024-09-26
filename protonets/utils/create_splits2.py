import os
import random

import os
import random
import argparse

import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/new_dataset/data')
SPLIT_DIR = os.path.join(os.path.dirname(__file__), '../../data/new_dataset/splits/new_split')

# Cria o diretório de splits se não existir
os.makedirs(SPLIT_DIR, exist_ok=True)

# Lista todas as classes (diretórios) no diretório de dados
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# Embaralha as classes para dividir aleatoriamente
random.shuffle(classes)

# Define proporções para cada split
train_ratio = 0.50
val_ratio = 0.30
test_ratio = 0.20

# Calcula o número de classes para cada split
n_total = len(classes)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

# Divide as classes em splits
train_classes = classes[:n_train]
val_classes = classes[n_train:n_train + n_val]
test_classes = classes[n_train + n_val:]

# Função para escrever classes em um arquivo
def write_split(classes, split_name):
    with open(os.path.join(SPLIT_DIR, f"{split_name}.txt"), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

# Escreve as classes em arquivos de split
write_split(train_classes, 'train')
write_split(val_classes, 'val')
write_split(test_classes, 'test')

# Cria e escreve o arquivo trainval.txt que une as classes de treinamento e validação
trainval_classes = train_classes + val_classes
write_split(trainval_classes, 'trainval')

print(f"Train classes: {len(train_classes)}")
print(f"Validation classes: {len(val_classes)}")
print(f"Test classes: {len(test_classes)}")
print(f"Train + Validation classes: {len(trainval_classes)}")



