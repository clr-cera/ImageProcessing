# Gatos, Cachorros e outros!

Esse trabalho tem o objetivo de explorar o uso de descritores visuais de imagem para fazer re-identificação de pets dos alunos da disciplinas de Processamento de Imagens;
e realizar busca de imagens similares à desejada

# Pre-Processamento!

Primeiro removi classes que tivessem apenas um exemplo e dividi as imagens entre treino, validação e teste.

O teste será apenas usado para avaliar os 3 melhores modelos de classificação.

# Descritores

## 1. Histograma Global

O Histograma global vai ser usado para analisar quão útil é a características globais de cor da imagem, principalmente em contraste com correlograma de cores que possui informação local.

## 2. Histograma Preto e Branco

O Histograma Preto e Branco serve para analisar o quanto de acurácia é perdida quando projetamos as cores em 3 canais para 1 canal de iluminação.

## 3. Correlograma de Cores

O Correlograma de Cores usa a distância entre pixeis para calcular a probabilidade de uma cor estar a uma certa distância da outra, ou seja é característica de cor com noção local!!
(Spoiler, foi o descritor que obteve maior acurácia em classificação)

## 4. Haralick

O descritor de haralick calcula várias estatísticas em cima da matriz de co ocorrência da imagem em preto e branco, por isso é um descritor com vários significados em textura.

## 5. Local Binary Patterns

O descritor LBP utiliza padrões para comparar um pixel com seus vizinhos e gerar um encoding binário, também é um descritor de textura

## 6. Histograma de Gradientes Orientados

O descritor HOG utiliza dos gradientes calculados pelo filtro Sobel e agrega baseado no ângulo, o que gera uma noção geral da angulação das bordas da imagem.

## 7. GIST
O GIST é o descritor que escolhi de fora da disciplina! É um descritor que utiliza vários filtros de Gabor (Que são baseado em noções biológicas de percepção de imagens), para analizar as textura em diferentes orientações, e a frequência que aparecem. Como ele é agregado em diferentes grids da imagem, também preserva a localidade das texturas. (Spoiler, ele foi o melhor descritor para busca! E conseguiu resultados ok para a classificação.) 

# Classificação
Utilizei 3 modelos para a classificação, um SVM, u KNN e um XGBoost, os modelos SVM e KNN são consideravelmente mais leves que o XGBoost, portanto espero que o XGBoost alcance acurácias maiores

## Resultados

### SVM

| Descritor | Acurácia (Validação) |
|---|---|
| Global Histogram | 16.22% |
| BW Histogram | 16.22% |
| Color Correlogram | **21.62%** |
| Haralick | 18.92% |
| Local Binary Patterns | 21.62% |
| Histogram of Oriented Gradients | 10.81% |
| GIST | 16.22% |
| Concatenated | 16.22% |

O melhor descritor foi o Correlograma de Cores com acurácia de teste: **33.33%**

### KNN

| Descritor | Melhor K | Acurácia (Validação) |
|---|---|---|
| Global Histogram | 1 | 24.32% |
| BW Histogram | 1 | 16.22% |
| Color Correlogram | 1 | **35.14%** |
| Haralick | 31 | 18.92% |
| Local Binary Patterns | 1 | 24.32% |
| Histogram of Oriented Gradients | 13 | 18.92% |
| GIST | 1 | 21.62% |
| Concatenated | 1 | 21.62% |

O melhor descritor foi o Correlograma de Cores de novo, com acurácia no teste: **30.56%**
Obteve uma acurácia maior que o SVM na validação mas pior no teste.

### XGBoost

| Descritor | Acurácia (Validação) |
|---|---|
| Global Histogram | 24.32% |
| BW Histogram | 18.92% |
| Color Correlogram | 27.03% |
| Haralick | 27.03% |
| Local Binary Patterns | 35.14% |
| Histogram of Oriented Gradients | 13.51% |
| GIST | 27.03% |
| Concatenated | **40.54%** |

O melhor resultado foi com todos os descritores concatenados, que alcançou acurácia no teste: **36.11%**
Essa foi a melhor acurácia no teste

### BoVW

| Modelo | Acurácia (Validação) | Acurácia (Teste) |
|---|---|---|
| SVM | 18.92% | 13.89% |
| KNN | 10.81% | 16.67% |
| XGBoost | 18.92% | 19.44% |


Também fiz a classificação usando Bag of Visual Words mas não performou bem :/

## Matrizes de Confusão

Calculei as matrizes de confusão para o melhor modelo de svm, knn e xgboost, que se encontram abaixo:

![Matriz de Confusão SVM](./svm_cm.png)
![Matriz de Confusão KNN](./knn_cm.png)
![Matriz de Confusão XGBoost](./xgboost_cm.png)

Todas as matrizes estavam bem distribuídas e todos erraram muito. Não sei o que analisar além disso.

# Busca

Para a tarefa de busca fiz uma função que recebe um descritor e uma imagem e busca as imagens no espaço daquele descritor. Testei a busca com a primeira imagem da Ada Pipoca.

![Busca da ADA Pipoca com descritor GIST](./gist_search.png)

O melhor resultado que tive foi utilizando o descritor GIST, em que encontrei duas outras Adas Pipocas.

Todos meus outros testes falharam em achar ADA Pipocas :(

![Busca da ADA Pipoca com descritor Haralick](./haralick_search.png)

Este é um exemplo utilizando o haralick, em que não consegui ver qual o motivo das imagens terem sido similares .

![Busca da ADA Pipoca com descritor LBP](./lbp_search.png)

Utilizando o LBP me pareceu que os pets estavam em posições parecidas, mas não tenho certeza.

E os resultados para os outros descritores também foram semelhantes. Acredito que consegui resultados melhores com GIST por ele ter uma visão geral da imagem baseado em percepção, foi de longe o descritor mais demorado para rodar e carrega mais informação que os outros descritores.

Também implementei Bag of Visual Words para esta tarefa, mas também não funcionou bem para a busca como pode ser visto abaixo :(.
![Busca da ADA Pipoca com descritor BOVW](./bovw.png)

# Representação da Bag of Visual Words
Abaixo estão ambos o UMAP e tSNE da bag of visual words das imagens. Todas estavam bem espalhadas o que explica o porque do modelo ter sido ruim para classificar e buscar.

![TSNE da bag of visual words das imagens](./bovw_umap.png)
![UMAP da bag of visual words das imagens](./bovw_tsne.png)

# Hipóteses
Acredito que o Correlograma de Cores foi melhor para a classificação porque ele possui a informação de cor que é muito importante para discernir as imagens, e a informação espacial permite a ele ver similaridade em que o pet está em lugares diferentes da imagem e com fundos diferentes. Os outros descritores não possuem ou a informação de cor ou a informação de localidade

Já na busca acredito que o GIST foi melhor por carregar mais informação das imagens e ser baseado numa ideia geral de percepção. Assim ele possui uma visão geral da imagem e também possui noção dos diferentes locais por causa do grid, e por isso alcançou melhores resultados. Os outros descritores eram mais leves e mais específicos.