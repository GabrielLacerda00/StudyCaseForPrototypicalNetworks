import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

from protonets.utils import filter_opt, merge_dict
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import torch

import torch
import numpy as np


def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = list(map(int, model_opt['model.x_dim'].split(',')))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = { 'data.' + k: v for k,v in filter_opt(model_opt, 'data').items() }

    print(data_opt)

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k,v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data, class_names = data_utils.load(data_opt, ['test'])

    print(class_names)

    if data_opt['data.cuda']:
        model.cuda()

    meters = { field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields'] }

    # Avaliação com coleta de métricas
    #meters, conf_matrix, class_report = model_utils.evaluate_metrics(model, data['test'], meters, desc="test")
    meters, conf_matrix, class_report = model_utils.plot_metrics(model, data['test'], meters,class_names,desc="test")

    # Exibindo a matriz de confusão
    print("Matriz de Confusão:")
    print(conf_matrix)

    # Exibindo o relatório de classificação
    print("\nRelatório de Classificação:")
    print(class_report)

    for field, meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(data_opt['data.test_episodes'])))