#  TitanicPredict  

##  Modelo de Machine Learning para prever a sobrevivência dos passageiros do Titanic  

**Dados:** [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)  

Um dos primeiros desafios clássicos do Kaggle é o Titanic Survival Prediction, cujo objetivo é prever se um passageiro sobreviveu ou não ao acidente, com base em informações como classe, idade, sexo, valor da passagem, entre outras variáveis.  

A ideia é treinar um modelo de Machine Learning utilizando os dados de treino e teste fornecidos, gerando uma classificação binária — 1 para sobreviveu, 0 para não sobreviveu.  

Neste projeto, utilizei as bibliotecas NumPy, Pandas e Scikit-Learn, aplicando o modelo Random Forest Classifier para realizar as predições.  

---

##  Etapas do Projeto  

### 1. Importação e Configuração Inicial  
Importei as bibliotecas necessárias e defini a semente (random_state=0) para padronizar os resultados e garantir reprodutibilidade.  
Em seguida, carreguei os arquivos train.csv e test.csv obtidos no Kaggle, salvando também o identificador do passageiro PassengerId para a planilha final de submissão.  

---

### 2. Análise das Variáveis  
Foi realizada uma análise exploratória para compreender quais variáveis possuíam maior influência na sobrevivência.  
As principais escolhidas foram:  

- **Numéricas:** Age, Fare (preço da passagem) e FamilySize (tamanho da família embarcada);  
- **Categóricas:** Pclass (classe do ticket), Sex, Embarked (porto de embarque), Title (título extraído do nome), Cabin (número da cabine) e isAlone (indicador de quem viajava sozinho).  

Busquei um equilíbrio entre **complexidade e interpretabilidade**, priorizando variáveis relevantes e com impacto direto no modelo.  

---

##  Pré-processamento  

### 🔹 Extração do Título (`Title`)  
A coluna Name continha o título do passageiro junto ao nome, separado por vírgula e ponto.  
Foi criada uma função para **extrair o título** (ex: Mr, Miss, Mrs, Dr).  

Ajustes adicionais:
- Substituí Mlle e Ms por Miss, e Mme por Mrs;  
- Agrupei títulos menos comuns na categoria "Rare", incluindo:  
  'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'.  

### 🔹 Criação de Novas Features  
- **FamilySize**: soma de SibSp (irmãos/cônjuges a bordo) + Parch (pais/filhos a bordo) + 1 (o próprio passageiro).  
- **isAlone**: variável binária que indica se o passageiro estava sozinho.  

### 🔹 Pipeline de Pré-processamento  
Foram criadas **pipelines separadas para variáveis numéricas e categóricas**:  
- Numéricas: substituição de valores ausentes pela **mediana**;  
- Categóricas: substituição pela **categoria mais frequente**, seguida de **codificação One-Hot** (OneHotEncoder).  

Essas transformações são aplicadas **apenas em memória**, sem modificar os arquivos originais, garantindo consistência e reprodutibilidade.  

---

##  Modelo de Machine Learning  

Utilizei o **RandomForestClassifier**, um modelo de aprendizado em conjunto (ensemble) composto por várias árvores de decisão.  
Cada árvore realiza previsões independentes, e o resultado final é determinado pelo voto majoritário das árvores.  

**Principais parâmetros utilizados:**
- `n_estimators=100` → número de árvores na floresta;  
- `max_depth=None` → profundidade máxima das árvores (ilimitada);  
- `min_samples_split=2` → número mínimo de amostras para dividir um nó;  
- `random_state=42` → garante a reprodutibilidade dos resultados.  

A avaliação foi feita com **validação cruzada (Cross-Validation)**, utilizando `StratifiedKFold` com `cv=5` divisões.  
Essa técnica permite medir o desempenho do modelo de forma mais confiável, evitando viés causado por um único corte de treino e teste.  

---

##  Resultados  

- **Acurácia na validação cruzada:** ~82%  
- **Acurácia na submissão do Kaggle:** 0.77  

Os resultados indicam uma boa capacidade de generalização do modelo, sem sinais significativos de overfitting.  

---


##  Aprendizados  

Durante o desenvolvimento deste projeto, consolidei conhecimentos sobre:
- Pré-processamento de dados (tratamento de nulos e codificação de variáveis categóricas);  
- Criação e manipulação de novas features;  
- Aplicação e avaliação de modelos de classificação supervisionada;  
- Uso de pipelines para um fluxo de tratamento reproduzível e limpo;  
- Entendimento prático da validação cruzada e importância da reprodutibilidade.  

---

##  Próximos Passos  

- Testar outros modelos como **XGBoost**, **Logistic Regression** e **Gradient Boosting**;  
- Otimizar hiperparâmetros com **GridSearchCV** ou **Optuna**;  
- Implementar **interpretação do modelo** com SHAP ou LIME;  
- Criar um **dashboard interativo** para visualização das previsões.  

---
