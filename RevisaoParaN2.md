# Inteligência Artificial - Revisão Para a N2

## Aprendizado Supervisionado
Trata de dois tipos de problemas ou tarefas: classificação ou regressão. É uma abordagem de aprendizado de máquina definida pelo uso de conjunto de dados rotulados. É o mapeamento do conjunto de dados de entrada (variáveis preditoras) e as respectivas categorias (rótulos) que podem ser então os novos casos para a predição.

* Regressão: predição de valores.
* Classificação: classificação em categorias.

## Aprendizado não supervisionado
Usa algorítmos de aprendizado de máquina para analisar e agrupar conjuntos de dados não rotulados. A idéia desses algorítmos é encontrar padrões ocultos nos dados **sem a necessidade de qualquer intervenção humana ou rótulos pré-definidos e, por isso, não supervisionados.**

## Modelos de Classificação
Modelos de Classificação estimam a partir dos atributos de entrada categorias. 

### Classificador Logístico (Regressão Logística)
A Regressão Logística modela as probabilidades para problemas de classificação binários, com dois resultados possíveis, como yes/no, true/false, fraude/não fraude, spam/não spam ou 0/1, e pode ser entendido como uma extensão dos modelos de regressão linear para problemas de classificação

**Importante: A Regressão Logística é um classificador Binário, isto é, ele só classifica categorias Dicotômicas, como yes/no, true/false!**

Para obter essa probabilidade a regressão logística buscará os melhores coeficientes em uma expressão semelhante a empregada em regressão linear. É uma expressão que emprega uma combinação linear das variáveis preditoras à qual aplicamos a função logística 𝜎(𝑥)=1/(1+𝑒^(−𝑥)) 

```
𝑝 = 1/(1+𝑒^(−(𝑎0+𝑎1𝑥1+...+𝑎𝑛𝑥𝑛))) 
```

### Métricas

#### Matriz de Confusão
A matriz de confusão é uma matriz que sumariza os resultados de acertos e erros do modelo para avaliar o desempenho da classificação. É uma matriz quadrada,  𝑛×𝑛  onde  𝑛  é o número de classes objetivo. Lembrando que avaliamos o modelo sobre os resultados no conjunto de teste, a matriz compara os valores reais (conjunto de teste) com aqueles estimados pelo modelo

#### Acuracidade
$$ Accuracy = \frac{Total de Acertos}{Total de Casos} $$

#### Precisão
A Precisão é um valor que, dados todos elementos previstos uma classe, quantos foram previstos corretamente. Isto é, o percentual dos casos que de fato pertencem àquela classe. 

$$ Precision = \frac{TP}{TP + FP} $$

#### Recall
O Recall (Revocação, ou Sensibilidade) por outro lado nos diz quantos casos de uma determinada classe foram corretamente previstos.

$$ Recall = \frac{TP}{TP + FN} $$

#### F1-Score
O F1-score pode ser entendido como uma média harmônica dos valores de precisão e recall:

$$ F1-score = \frac{2}{1/Recall + 1/Precision}$$

Na prática, quando tentamos aumentar a precisão do nosso modelo, o recall diminui e vice-versa. A pontuação F1 permite capturar ambas as tendências em um único valor e, por isso é bastante empregada sendo seu valor máximo quando a precisão e o recall são iguais.

#### Classification Report
Todos conceitos acima são importantes para entendermos as métricas, mas todas essas métricas são mais facilmente obtidas no `classification_report`.

```
Classification Report:

              precision    recall  f1-score   support

           0       0.69      1.00      0.81        24
           1       1.00      0.08      0.15        12

    accuracy                           0.69        36
   macro avg       0.84      0.54      0.48        36
weighted avg       0.79      0.69      0.59        36
```

### K-Vizinhos mais Próximos
Os K-Vizinhos mais Próximos, ou KNN (do inglês, K nearest neighbors) é um dos modelos mais simples de classificação, mas também bastante empregado. Seu funcionamento se baseia na classificação de uma instância de acordo com a classe de seus vizinhos mais próximos. O número k define quantos vizinhos queremos empregar na classificação.

O conceito do Knn é bastante simples o que permite implementar o algoritmo e verificar o seu funcionamento sem qualquer API ou pacote adicional. Basicamente o modelo consiste na execução de 3 passos:
* Calcular as distâncias do elemento desejado para os demais
* Encontrar os k-vizinhos mais próximos
* Retornar a classe mais frequente entre dos k-vizinhos

#### Normalizando os Dados
O cálculo de distâncias como medida de similaridade (menor distância indicando maior similaridade) pode, entretanto, apresentar grandes desvios quando empregamos variáveis ​com escalas muito diferentes ou variáveis ​​numéricas e categóricas em conjunto. 

Vários modelos de aprendizado de máquina são baseados em distância como medida de similaridade e são, portanto, sensíveis à normalização dos dados e devemos aplicá-la quando empregados dados em diferentes escalas.

Você pode simplesmente empregar uma função `minmax_scale` do `scikit-learn` para fazer a normalização, mas existem outras funções para os demais tipos de normalização.

```py
from sklearn.preprocessing import minmax_scale

minmax_scale(loans[['age','loan']])
```
```
array([[0.125     , 0.10891089],
       [0.375     , 0.20792079],
       [0.625     , 0.30693069],
       [0.        , 0.00990099],
       [0.375     , 0.5049505 ],
       [0.8       , 0.        ],
       [0.075     , 0.38118812],
       [0.5       , 0.21782178],
       [1.        , 0.40594059],
       [0.7       , 1.        ],
       [0.325     , 0.65346535]])
```

#### Exemplo
```py
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

loans = pd.DataFrame({'age':[25,35,45,20,35,52,23,40,60,48,33],
                      'loan':[40000,60000,80000,20000,120000,18000,95000,62000,100000,220000,150000],
                      'default':[1,1,1,1,1,1,0,0,0,0,0] }) # 1='yes'

case  = pd.DataFrame({'age':[47],'loan':[142000]})

X = loans[['age','loan']]      
y = loans.default   

scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X) 
case_scaled = scaler.transform(case)

clf = neighbors.KNeighborsClassifier(n_neighbors = 3)

clf.fit(X, y)                 

y_pred = clf.predict(case_scaled)

default_pred = ['No','Yes'][y_pred[0]]
print('Default? ', default_pred)
```

### Árvore de Decisão e o Método Partitivo
Uma estrutura de árvore 'modela' os dados e divide recursivamente o espaço em regiões com classes semelhantes. Essa estrutura faz particionamentos sucessivos dos dados em que as partições são cada vez mais puras, isto é, são cada vez mais homogêneas no sentido das classes que contêm.

![ArvoreDecisao](https://github.com/Rogerio-mack/BIG_DATA_Analytics_Mineracao_e_Analise_de_Dados/blob/main/figuras/DecisionTreePartitions.png?raw=true)

O nó raiz da árvore representa todo o conjunto de dados. Este conjunto é então dividido aproximadamente ao meio ao longo de uma dimensão (um atributo) por um limite simples. Todos os pontos que possuem um valor  >𝑡  caem no nó filho direito e todos os demais no nó filho esquerdo. O limiar  𝑡  a dimensão é escolhida de forma que os nós filhos resultantes sejam mais puros em termos de associação de classe. O ideal é que todos os pontos positivos (por exemplo, pontos laranja na figura acima) caiam em um nó filho e todos os pontos negativos (roxos) no outro. Se for esse o caso, a árvore está pronta. Caso contrário, os nós folha são novamente divididos até que eventualmente todas as folhas, ou nós terminais, sejam puras com todos seus pontos com o mesmo rótulo ou não podem ser mais divididos (dois pontos idênticos com rótulos diferentes, o que em casos reais podemos de fato ter).

Uma vez que a árvore é construída os dados de treinamento não precisam ser mais armazenados (como no Knn) pois a árvore captura todo o padrão dos dados, e você pode pensar em como uma única fórmula  𝑦=𝑎0+𝑎1𝑥  modela um conjunto de dados em uma regressão linear. A árvore passa a ser nossa 'fórmula' de predição, as entradas de teste simplesmente precisam descer da árvore até uma folha que contendo a predição da classe, e são muito eficientes. Além disso as árvores de decisão não requerem métricas porque as divisões são baseadas em limites dos valores dos atributos (ou ainda suas proporções ou probabilidades) e não em distâncias.

Agora, um exemplo de como construir uma árvore de decisão a partir dos dados:

| Name     | sex    | smokes | tie | mask | cape | ears | class |
|----------|--------|--------|-----|------|------|------|-------|
| Batman   | male   | no     | no  | yes  | yes  | yes  | good  |
| Robin    | male   | no     | no  | yes  | yes  | yes  | good  |
| Catwoman | female | no     | no  | yes  | no   | yes  | bad   |
| Joker    | male   | no     | no  | no   | no   | no   | bad   |
| Alfred   | male   | no     | yes | no   | no   | no   | good  |
| Penguin  | male   | yes    | yes | no   | no   | no   | bad   |

| Name     | sex    | smokes | tie | mask | cape | ears | class |
|----------|--------|--------|-----|------|------|------|-------|
| Batgirl  | female | yes    | yes | no   | yes  | no   | ?     |
| Riddler  | male   | yes    | no  | no   | no   | no   | ?     |

O algoritmo de Hunt cria uma árvore de decisão de maneira recursiva, particionando os registros de treinamento em subconjuntos sucessivamente mais puros. Seja $D_t$ o conjunto de registros de treinamento que atingem um nó $t$. O procedimento recursivo é, então, o seguinte:

1. Se $D_t$ contém registros que pertencem à mesma classe $y_t$, então $t$ é um nó folha rotulado como $y_t$

2. Se $D_t$ for um conjunto vazio, então $t$ é um nó folha rotulado pela classe padrão, $y_d$

3. Se $D_t$ contiver registros que pertencem a mais de uma classe, use um teste de atributo para dividir os dados em subconjuntos menores.

Ele aplica recursivamente o procedimento a cada subconjunto até que todos os registros no subconjunto pertençam à mesma classe. O algoritmo de Hunt assume que cada combinação de conjuntos de atributos possui um rótulo de classe exclusivo durante o procedimento. Se todos os registros associados a $D_t$ tiverem valores de atributo idênticos, exceto para o rótulo da classe, não será possível dividir esses registros no futuro. Nesse caso, o nó é classificado como um nó folha com o mesmo rótulo de classe que a classe principal de registros de treinamento associados a este nó.

![ArvoreFuncionando](https://github.com/Rogerio-mack/BIG_DATA_Analytics_Mineracao_e_Analise_de_Dados/blob/main/figuras/DecisionTree3.png?raw=true)

> *Note: Árvores de Decisão não requerem valores numéricos e podem empregar valores categóricos diretamente. Entretanto, o estimador do scikit-learn não está implementado deste modo e requer que os valores dos atributos sejam numéricos. Por isso, é necessário fazer o encode dos dados antes de aplicar o estimador.*

#### Entropia e Ganho de Informação
A escolha de atributos para a definição dos nós da árvore segue o critério de empregar os atributos com maior ganho de informação para maximizar a pureza dos particionamentos. Você pode entender o ganho de informação como uma medida de quanto a informação de um atributo contribui para diminuir a incerteza sobre outro.

A *Entropia* de um atributo é o nível médio de informação, surpresa, ou ainda incerteza, inerente aos resultados possíveis desse atributo.

O *Ganho de Informação* de um atributo com relação a uma variável classe objetivo ( 𝑇 , Target) é uma medida de quanto a informação desse atributo diminui a incerteza sobre a classe dos dados, ou seu valor no caso de uma regressão.

## Aprendizado Não Supervisionado
São muitas as situações em que não temos a possibilidade de termos dados de treinamento pré-rotulados. Apesar disso, queremos extrair conhecimentos úteis dos dados para a tomada de decisões e ações. É nesta situação que modelos de aprendizado não supervisionado são bastante úteis. Não havendo um conjunto de entradas e saídas, isto é um conjunto de treinamento, o objetivo do aprendizado não supervisionado, será o de descobrir padrões nos dados. O modelo ou algoritmo irá tentar aprender estruturas, relações e padrões latentes nos dados sem qualquer assistência ou supervisão.

Algoritmos de aprendizado não supervisionado normalmente incluem: 

* Algoritmos de clusterização (ou agrupamento) 
* Detecção de anomalias
* Redução de Dimensionalidade (ou Métodos de Variáveis Latentes) 
* Regras de Associação
       
Os métodos de clusterização incluem clusterização hierárquica, k-médias. Para detecção de anomalias há métodos como Fator Outlier Local e Floresta de Isolamento. A análise de componentes principais (PCA) e valor singular decomposição (SVD) são métodos empregados para redução de dimensionalidade e algoritmos *apriori* são aplicados para busca de regras de associação.

Em todos esses casos há certamente padrões particulares nos dados que aparecem com mais frequência e outros menos, e queremos descobrir o que acontece e o que não acontece nos dados. Isso, em estatística, denominado de estimativa de densidade.

### Aprendizado Supervisionado vs Não Supervisionado
