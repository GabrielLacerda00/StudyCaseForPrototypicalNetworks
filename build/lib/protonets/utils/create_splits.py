import os
import random
import argparse

def create_splits(data_dir, split_dir, split_name='vinyals', train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    """
    Cria arquivos de split para treinamento, validação e teste.

    Args:
        data_dir (str): Diretório contendo as classes do dataset.
        split_name (str): Nome do split (ex: 'vinyals', 'new_split').
        train_ratio (float): Proporção de classes para treinamento.
        val_ratio (float): Proporção de classes para validação.
        test_ratio (float): Proporção de classes para teste.

    Raises:
        ValueError: Se a soma das proporções não for 1.
    """
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("A soma das proporções de treinamento, validação e teste deve ser 1.")

    # Listar todas as classes (subdiretórios)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Embaralhar as classes para garantir uma divisão aleatória
    random.shuffle(classes)
    
    # Calcular quantidades para cada split
    total_classes = len(classes)
    train_size = int(total_classes * train_ratio)
    val_size = int(total_classes * val_ratio)
    test_size = total_classes - train_size - val_size

    # Dividir as classes
    train_classes = classes[:train_size]
    val_classes = classes[train_size:train_size + val_size]
    test_classes = classes[train_size + val_size:]

    # Função para obter exemplos de arquivos
    def get_example_files(cls):
        class_dir = os.path.join(data_dir, cls)
        examples = []
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith('.jpg'):  # Supondo que os arquivos de exemplo são imagens JPG
                    examples.append(os.path.join(cls, file))
        return examples

    # Criar o diretório de splits, se não existir
    splits_dir = os.path.join(split_dir, 'splits', split_name)
    os.makedirs(splits_dir, exist_ok=True)
    
    # Salvar as classes e exemplos de arquivos em arquivos de texto
    def save_split(file_path, classes):
        with open(file_path, 'w') as f:
            for cls in classes:
                examples = get_example_files(cls)
                for example in examples:
                    f.write(f"{example}\n")

    save_split(os.path.join(splits_dir, 'train.txt'), train_classes)
    save_split(os.path.join(splits_dir, 'val.txt'), val_classes)
    save_split(os.path.join(splits_dir, 'test.txt'), test_classes)
    
    print(f"Splits '{split_name}' criados com sucesso!\nTreinamento: {train_size} classes\nValidação: {val_size} classes\nTeste: {test_size} classes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cria splits para treinamento, validação e teste.')
    parser.add_argument('--data_dir', type=str, required=True, help='Diretório contendo as classes do dataset.')
    parser.add_argument('--split_dir', type=str, required=True, help='Diretório onde os arquivos de split serão salvos.')
    parser.add_argument('--split_name', type=str, default='vinyals', help='Nome do split (default: vinyals).')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Proporção de classes para treinamento (default: 0.5).')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='Proporção de classes para validação (default: 0.25).')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='Proporção de classes para teste (default: 0.25).')
    args = parser.parse_args()

    create_splits(args.data_dir, args.split_dir, args.split_name, args.train_ratio, args.val_ratio, args.test_ratio)
