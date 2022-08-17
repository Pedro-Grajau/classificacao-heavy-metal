### Heavy Metal Classificator :metal:
---
Aplicação para classificação de músicas Heavy Metal utilizando regressão logística e TFidfvectorizer <br>
Este projeto foi uma ideia da comunidade de challenges da Let's Code: https://discord.com/invite/khJTEK7bsY <br>
Base de Dados: https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?select=artists-data.csv

![image](https://user-images.githubusercontent.com/84031272/185224284-1d1c0125-addc-4fc0-82e4-9791f8e20944.png)

### Bibliotecas Utilizadas <br>

> - pandas (Para manipulação dos datasets)
> - nltk (Para o pré-processamento dos textos)
> - sklearn (Para criação do modelo de ml)
> - streamlit (Para criação da aplicação web)
> - matplotlib (Para criação do gráfico)
> - seaborn (Para criação da matriz de confusão)

### Etapas Desenvolvidas

- #### Carregamento e tratamento do Dataset 

Nessa etapa foi feita uma escolha das colunas relevantes para o dataset até então, depois de juntar os dois datasets do kaggle (artists, lyrics). Depois foram removidos os valores nulos.

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 364375 entries, 0 to 378987
Data columns (total 5 columns):
 #   Column    Non-Null Count   Dtype 
---  ------    --------------   ----- 
 0   SName     364375 non-null  object
 1   Lyric     364375 non-null  object
 2   language  364375 non-null  object
 3   Artist    364375 non-null  object
 4   Genres    364375 non-null  object
dtypes: object(5)
memory usage: 16.7+ MB
```

- #### Escolha da linguagem da música (en, pt) e do gênero escolhido para predição 

![image](https://user-images.githubusercontent.com/84031272/185226905-431df69e-276c-4a1f-b467-3542869200a2.png) <br>
Nessa etapa optou-se por escolher Heavy Metal por ser um gênero mais específico e bastante predominante na linguagem escolhida (en)

- #### Processamento de texto utilizando as seguintes técnicas
  - Tokenização
  - Remoção de stopwords
  - Lemmatization
  - Vetorização

A vetorização foi feita usando o TFidfvectorizer, por ser um método bastante tradicional em pré-processamento de texto
```
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.8)
X_tfidf_vectorizer = tfidf_vectorizer.fit_transform(df_train['Lyric']).toarray()
```
  
- #### Criação do modelo de regressão logística 
Para balancear o modelo, foi criado um dataset com 40.000 entradas, sendo metade delas de músicas que são Heavy Metal e a outra metade que não são. Isso me garante que haverá um treinamento justo do modelo.

```
genre = "Heavy Metal"

def get_target_number(element):
    return 1 if genre in element else 0
    
df['target'] = df['Genres'].apply(get_target_number)

positive = df[df['target'] == 1].sample(n=20000, random_state=1)
negative = df[df['target'] == 0].sample(n=20000, random_state=1)

df_train = pd.concat([positive, negative])
```
Dessa forma, utilizando o modelo de Regressão Logística foi obtido score e precisão de 80%, o que é um valor já bem interessante visto que não foi adotado nenhum tipo de tuning no modelo.

![image](https://user-images.githubusercontent.com/84031272/185232322-ea2bc8f6-69ed-4d47-bd3f-08e32b3862b0.png)

- #### Criação da aplicação web
Utilizando streamlit é possível passar a música que deseja testar por um upload de um txt que é a letra da música em questão

![image](https://user-images.githubusercontent.com/84031272/185233066-53b32c8c-848d-46f3-a681-5be09e93d0ad.png)


