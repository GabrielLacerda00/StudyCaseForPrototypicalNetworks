# Estudo de caso do uso de Redes Neurais Prototípicas para Few-Shot Learning na classificação de imagens de documentos com desbalanceamento
 
Este código é uma adaptação do código utilizado no artigo "Prototypical Networks for Few Shot Learning" ([paper](https://arxiv.org/abs/1703.05175), [code](https://github.com/jakesnell/prototypical-networks)). Este repositório contém os resultados das avaliações utilizando Few-Shot Learning em um modelo de Redes Prototípicas para classificação de documentos. Foram realizados experimentos com diferentes configurações, utilizando uma estrutura de 5 classes e variação no número de exemplos e consultas por classe. 

Este experimento foi realizado em um ambiente Ubuntu 22.04.4 LTS com uma GeForce RTX 4090.


## Few-Shot Learning - Resultados dos Experimentos - Relatórios de Classificação

### 1. Avaliação: 5-way, 1-shot com 5 consultas por classe (1000 episódios)

| Classe                         | Precisão | Recall | F1-Score | Suporte |
|---------------------------------|----------|--------|----------|---------|
| CUPOM-FISCAL                    | 0.76     | 0.75   | 0.75     | 5000    |
| RECEITA-VERSO                   | 0.72     | 0.74   | 0.73     | 5000    |
| CERTIDÃO-NASCIMENTO             | 0.75     | 0.76   | 0.75     | 5000    |
| REGISTRO-NACIONAL-ESTRANGEIRO-VERSO | 0.75     | 0.74   | 0.74     | 5000    |
| CNH2                            | 0.75     | 0.73   | 0.74     | 5000    |

**F1-Score Médio**: 0.74  
**Acurácia**: 0.744 +/- 0.0065  
**Perda (Loss)**: 0.695 +/- 0.0203


---

### 2. Avaliação: 5-way, 5-shot com 5 consultas por classe (1000 episódios)

| Classe                         | Precisão | Recall | F1-Score | Suporte |
|---------------------------------|----------|--------|----------|---------|
| CUPOM-FISCAL                    | 0.88     | 0.88   | 0.82     | 5000    |
| RECEITA-VERSO                   | 0.87     | 0.87   | 0.82     | 5000    |
| CERTIDÃO-NASCIMENTO             | 0.87     | 0.82   | 0.82     | 5000    |
| REGISTRO-NACIONAL-ESTRANGEIRO-VERSO | 0.87     | 0.83   | 0.83     | 5000    |
| CNH2                            | 0.81     | 0.81   | 0.81     | 5000    |

**F1-Score Médio**: 0.87  
**Acurácia**: 0.874 +/- 0.0043  
**Perda (Loss)**: 0.403 +/- 0.0112


---

### 3. Avaliação: 5-way, 1-shot com 15 consultas por classe (600 episódios)

| Classe                         | Precisão | Recall | F1-Score | Suporte |
|---------------------------------|----------|--------|----------|---------|
| CUPOM-FISCAL                    | 0.74     | 0.71   | 0.72     | 9000    |
| RECEITA-VERSO                   | 0.69     | 0.71   | 0.70     | 9000    |
| CERTIDÃO-NASCIMENTO             | 0.72     | 0.72   | 0.72     | 9000    |
| REGISTRO-NACIONAL-ESTRANGEIRO-VERSO | 0.70     | 0.72   | 0.71     | 9000    |
| CNH2                            | 0.71     | 0.70   | 0.71     | 9000    |

**F1-Score Médio**: 0.71  
**Acurácia**: 0.711 +/- 0.0074  
**Perda (Loss)**: 0.825 +/- 0.0280


---

### 4. Avaliação: 5-way, 5-shot com 15 consultas por classe (600 episódios)

| Classe                         | Precisão | Recall | F1-Score | Suporte |
|---------------------------------|----------|--------|----------|---------|
| CUPOM-FISCAL                    | 0.86     | 0.86   | 0.86     | 9000    |
| RECEITA-VERSO                   | 0.86     | 0.86   | 0.86     | 9000    |
| CERTIDÃO-NASCIMENTO             | 0.86     | 0.86   | 0.86     | 9000    |
| REGISTRO-NACIONAL-ESTRANGEIRO-VERSO | 0.86     | 0.85   | 0.85     | 9000    |
| CNH2                            | 0.85     | 0.86   | 0.85     | 9000    |

**F1-Score Médio**: 0.86  
**Acurácia**: 0.856 +/- 0.0040  
**Perda (Loss)**: 0.487 +/- 0.0142


---

## Como Reproduzir
1. Crie um ambiente Python:
   ```bash
   python -m venv meu_ambiente
   source meu_ambiente/bin/activate  # Para Linux/Mac
   ```

2. Instale as dependências e faça os imports necessários:
   ```bash
   pip install -r requirements.txt
   ```

3. Aplique o split na base de dados:
   ```bash
   python create_splits.py
   ```

4. Execute o treinamento e teste do modelo:
   ```bash
   python3 scripts/train/few_shot/run_train_test.py --data.cuda
   ```

5. Execute o modelo no modo de treino e avaliação:
   ```bash
   /home/lpc/Downloads/prototypical-networks/prototypical-networks/comands.txt
   ```

6. Rode a avaliação do modelo:
   ```bash
   python3 scripts/predict/few_shot/run_evalmetrics.py
   ```

Seguindo esses passos, você conseguirá reproduzir os experimentos apresentados.

---

## .bib citation
cite the paper as follows (copied-pasted it from arxiv for you):

    @article{DBLP:journals/corr/SnellSZ17,
      author    = {Jake Snell and
                   Kevin Swersky and
                   Richard S. Zemel},
      title     = {Prototypical Networks for Few-shot Learning},
      journal   = {CoRR},
      volume    = {abs/1703.05175},
      year      = {2017},
      url       = {http://arxiv.org/abs/1703.05175},
      archivePrefix = {arXiv},
      eprint    = {1703.05175},
      timestamp = {Wed, 07 Jun 2017 14:41:38 +0200},
      biburl    = {http://dblp.org/rec/bib/journals/corr/SnellSZ17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
