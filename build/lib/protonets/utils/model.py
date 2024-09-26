from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import numpy as np

def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters


def evaluate_metrics(model, data_loader, meters, desc=None):
    model.eval()

    y_true = []
    y_pred = []

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)

        # Coletando as previsões e os rótulos verdadeiros
        y_true.extend(output['target'].flatten()) #.cpu().numpy()
        y_pred.extend(output['pred'].flatten()) #.cpu().numpy()

        for field, meter in meters.items():
            meter.add(output[field])

    # Gerando a matriz de confusão
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Gerando o relatório de classificação
    class_report = classification_report(y_true, y_pred)

    return meters, conf_matrix, class_report



def plot_metrics(model, data_loader, meters, class_names=None, desc=None):
    model.eval()

    y_true = []
    y_pred = []

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)

        # Coletando as previsões e os rótulos verdadeiros diretamente e achatando os arrays
        y_true.extend(output['target'].flatten())
        y_pred.extend(output['pred'].flatten())

        for field, meter in meters.items():
            meter.add(output[field])


    # Gerando a matriz de confusão
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Normalizando a matriz de confusão para porcentagens
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Gerando o relatório de classificação com os nomes das classes
    if class_names:
        class_report = classification_report(y_true, y_pred, target_names=class_names)
    else:
        class_report = classification_report(y_true, y_pred)

    # Calculando o F1-score para cada classe
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred)

    # Calculando o F1-score geral (ponderado)
    _, _, f1_score_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Exibindo o F1-score para cada classe
    print("\nF1-Score por classe:")
    if class_names:
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {f1_score[i]:.2f}")
    else:
        for i in range(len(f1_score)):
            print(f"Classe {i}: {f1_score[i]:.2f}")

    # Exibindo o F1-score geral
    print(f"\nF1-Score Geral (Ponderado): {f1_score_weighted:.2f}")

    # Plotando a matriz de confusão em porcentagem
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix in Percentage')

    # Ajustando a rotação dos labels
    plt.xticks(rotation=45, ha='right')  # Rotaciona os labels do eixo x
    plt.yticks(rotation=0)  # Mantém os labels do eixo y na horizontal

    # Ajustando o layout para não cortar os labels
    plt.tight_layout()


    # Salvando a matriz de confusão como imagem
    plt.savefig('confusion_matrix_percentage.png')
    plt.close()

    return meters, conf_matrix, class_report
