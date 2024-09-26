
import os
import glob
from functools import partial
import numpy as np
from PIL import Image
import torch
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from torch.utils.data import DataLoader
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

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
    cls = d['class']
    if cls not in cache:
        class_dir = os.path.join(data_dir, cls)
        class_images = sorted(glob.glob(os.path.join(class_dir, '*.jpg')))
        if len(class_images) == 0:
            raise Exception(f"No images found for class {cls} at {class_dir}.")

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

        loader = DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            cache[cls] = sample['data']
            break

    return { 'class': cls, 'data': cache[cls] }

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

def load_data(opt, splits):
    data_dir = 'data/new_dataset/data'
    split_dir = 'data/new_dataset/splits'
    split_name = 'new_split'
    ret = {}
    cache = {}

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

        split_file = os.path.join(split_dir,split_name, f"{split}.txt")
        class_names = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('/')
                class_name = parts[0]
                file_name = os.path.join(*parts[1:])
                class_names.append({'class': class_name, 'file_name': os.path.join(data_dir, class_name, file_name)})
        
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        ret[split] = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret

