import os
import random
import glob
from functools import partial
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from torch.utils.data import DataLoader
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler


def create_splits(data_dir, split_name='vinyals', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
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
    
    # Criar o diretório de splits, se não existir
    splits_dir = os.path.join(data_dir, 'splits', split_name)
    os.makedirs(splits_dir, exist_ok=True)
    
    # Salvar as classes em arquivos de texto
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for class_name in train_classes:
            f.write(class_name + '\n')

    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        for class_name in val_classes:
            f.write(class_name + '\n')

    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for class_name in test_classes:
            f.write(class_name + '\n')
    
    print(f"Splits '{split_name}' criados com sucesso!\nTreinamento: {train_size} classes\nValidação: {val_size} classes\nTeste: {test_size} classes")



def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

def load_class_images(data_dir, d, cache):
    if d['class'] not in cache:
        class_dir = os.path.join(data_dir, d['class'])

        class_images = sorted(glob.glob(os.path.join(class_dir, '*.png')))
        if len(class_images) == 0:
            raise Exception("No images found for class {} at {}.".format(d['class'], class_dir))

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

        loader = DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            cache[d['class']] = sample['data']
            break

    return { 'class': d['class'], 'data': cache[d['class']] }

def extract_episode(n_support, n_query, d):
    n_examples = d['data'].size(0)
    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load_data(data_dir, opt, splits):
    ret = {}
    cache = {}
    split_name = opt['data.split']
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_images, data_dir, cache),
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        split_file = os.path.join(data_dir, 'splits', split_name, f"{split}.txt")
        class_names = []
        with open(split_file, 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        ret[split] = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
