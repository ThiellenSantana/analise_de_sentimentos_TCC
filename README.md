# analise_de_sentimentos_TCC
#importar documento para leitura#
from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import io

arquivo = pd.read_excel('comentarios.xlsx')
arquivo.head()

#nulos = pd.DataFrame(arquivo)
#enulo = nulos.isnull().sum(0)
#print(enulo)

#importaçao train test #
from sklearn.model_selection import train_test_split
treino, teste, classe_treino, classe_teste = train_test_split (arquivo.comentario, 
                                                               arquivo.importancia,
                                                               random_state = 42)
                                                               
                                                               
print(arquivo.importancia.value_counts ())
#contador 

#treimando o modelo #
from sklearn.feature_extraction.text import CountVectorizer 
texto = ["Nossa que perigo", "Nossa nada a ver"]

vetorizar = CountVectorizer(lowercase=False)
bag_of_words = vetorizar.fit_transform (texto)

vetorizar.get_feature_names() 

#imprimindo o tamanho do vetor, selecionando somente as palavras que mais aparecem e delimitando um máximo de 50 palavras (max_features)
vetorizar = CountVectorizer(lowercase=False, max_features=50)
bag_of_words = vetorizar.fit_transform (arquivo.comentario)
print (bag_of_words.shape)

import nltk
nltk.download("all")

from sklearn.linear_model import LogisticRegression

#regressão logística para verificar acurácia

treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words, arquivo.importancia, test_size = 0.33, random_state = 42)

regressao_logistica = LogisticRegression()
regressao_logistica.fit(treino, classe_treino)
acuracia = regressao_logistica.score(teste, classe_teste)
print(acuracia)

#acurácia melhorada com algoritmo dentro de uma função 

from sklearn.linear_model import LogisticRegression

def classificar_texto(texto, coluna_texto, coluna_importancia):
  vetorizar = CountVectorizer(lowercase=False, max_features=50)
  bag_of_words = vetorizar.fit_transform(texto[coluna_texto])
  treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words, arquivo.importancia, test_size = 0.33, random_state = 42)

  regressao_logistica = LogisticRegression()
  regressao_logistica.fit(treino, classe_treino)
  return regressao_logistica.score(teste, classe_teste)
  
  !pip install wordcloud
  
  #nuvem de palavras 
from os import path
%matplotlib inline
from wordcloud import WordCloud

todas_palavras = ' '.join([texto for texto in arquivo.comentario])

nuvem_palavras = WordCloud(width= 800, height= 500,
                           max_font_size= 110,
                           collocations= False ).generate(todas_palavras)
                           
#converter a saída da nuvem de palavras em imagem usando o matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10,7))
plt.imshow(nuvem_palavras, interpolation='bilinear')
plt.axis("off")
plt.show

#separar os comentários preocupados 

def nuvem_palavras_s(texto, coluna_texto):
  texto_preocupado = texto.query("importancia == 'S'")
  todas_palavras = ' '.join([texto for texto in texto_preocupado[coluna_texto]])

  nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)

  plt.figure(figsize=(10,7))
  plt.imshow(nuvem_palavras, interpolation='bilinear')
  plt.axis("off")
  plt.show
  
  nuvem_palavras_s(arquivo, "comentario")
  
  #vamos separar os comentarios negativos 

def nuvem_palavras_neg(texto, coluna_texto):
  texto_negativo = texto.query("importancia == 'N'")
  todas_palavras = ' '.join([texto for texto in texto_negativo[coluna_texto]])

  nuvem_palavras = WordCloud(width=800, height=500, max_font_size=110, collocations=False).generate(todas_palavras)

  plt.figure(figsize=(10,7))
  plt.imshow(nuvem_palavras, interpolation='bilinear')
  plt.axis("off")
  plt.show
  
  nuvem_palavras_neg(arquivo, "comentario")
  
  #tokenizando o nosso texto:
from nltk import tokenize

token_espaco = tokenize.WhitespaceTokenizer()
todas_palavras = ' '.join([texto for texto in arquivo.comentario])
token_frase = token_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_frase)
df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), "Frequência": list(frequencia.values())})

df_frequencia.nlargest(columns = "Frequência", n = 10)

pip install --user -U nltk

#tratamento das frases sem as palavras irrelevantes 
palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

frase_processada = list()
for opiniao in arquivo.comentario:
  nova_frase = list()
  palavra_texto = token_espaco.tokenize(opiniao)
  for palavra in palavra_texto:
    if palavra not in palavras_irrelevantes:
      nova_frase.append(palavra)
  frase_processada.append(' '.join(nova_frase))

arquivo["tratamento_sempi"] = frase_processada

arquivo.head()

#validar melhora da acuracia
classificar_texto(arquivo, "tratamento_sempi", "importancia")

from string import punctuation
token_pontuacao = tokenize.WordPunctTokenizer()
pontuacao = list ()
for ponto in punctuation:
   pontuacao.append(ponto)
#incluir pontuação na lista de stopwords 
pontuacao_stopwords = pontuacao + palavras_irrelevantes

frase_processada = list()
for opiniao in arquivo["tratamento_sempi"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

arquivo["tratamento_sempont"] = frase_processada     

arquivo.head()

#validar melhora da acuracia
classificar_texto(arquivo, "tratamento_sempont", "importancia")

!pip install unidecode

#retirar acentos 
import unidecode
sem_acentos = [unidecode.unidecode(texto) for texto in arquivo ["tratamento_sempont"]]

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

arquivo["tratamento_semacen"] = sem_acentos

frase_processada = list()
for opiniao in arquivo["tratamento_semacen"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

arquivo["tratamento_semacen"] = frase_processada 

arquivo.head()

from sklearn.linear_model import LogisticRegression

regressao_logistica = LogisticRegression()

acuracia_tratamento = classificar_texto(arquivo, "tratamento_semacen", "classificacao")
print(acuracia_tratamento)

#stemmer para deixar somente a raiz da palavra

stemmer = nltk.RSLPStemmer()

frase_processada = list()
for line in arquivo["tratamento_semacen"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(stemmer.stem(word))
    frase_processada.append(' '.join(nova_frase))

arquivo["tratamento_raiz"] = frase_processada  

arquivo.head()

print(classificar_texto(arquivo, "tratamento_semacen", "importancia"))
print(classificar_texto(arquivo, "tratamento_raiz", "importancia"))

#transformando os dados em letras minúsculas

frase_processada = list()
for line in arquivo["tratamento_raiz"]:
    nova_frase = list()
    line = line.lower()
    palavras_texto = token_pontuacao.tokenize(line)
    for word in palavras_texto:
        if word not in pontuacao_stopwords:
            nova_frase.append(word)
    frase_processada.append(' '.join(nova_frase))

arquivo["tratamento_mins"] = frase_processada  

arquivo.head()

print(classificar_texto(arquivo, "tratamento_mins", "importancia"))

# tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_tratados = tfidf.fit_transform(arquivo["tratamento_mins"])
treino, teste, classe_treino, classe_teste = train_test_split(tfidf_tratados, arquivo["importancia"], random_state=42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_tratados = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_tratados)

#ngrams


tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
vetor_tfidf = tfidf.fit_transform(arquivo["tratamento_mins"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, arquivo["importancia"], random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print(acuracia_tfidf_ngrams)

#verificar os pesos positivos que o algoritimo está considerando
pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names()
)

pesos.nlargest(10, 0)

pesos.nsmallest(10,0)

#quais palavras aparecem mais
#está usando o código já com tratamento
import seaborn as sns

def pareto(texto, coluna_texto, quantidade):
  todas_palavras = ' '.join([arquivo for arquivo in texto[coluna_texto]])
  token_espaco = nltk.tokenize.WhitespaceTokenizer()
  token_frase = token_espaco.tokenize(todas_palavras)
  frequencias = nltk.FreqDist(token_frase)
  df_frequencias = pd.DataFrame({'Palavras': list(frequencias.keys()),
                               'Frequencia': list(frequencias.values())})
  df_frequencias = df_frequencias.nlargest(n=quantidade, columns='Frequencia')

  total = df_frequencias['Frequencia'].sum()
  df_frequencias['Porcentagem'] = df_frequencias['Frequencia'].cumsum() / total * 100

  plt.figure(figsize=(12,8))
  ax = sns.barplot(data=df_frequencias, x='Palavras', y='Frequencia', color='gray')
  ax2 = ax.twinx()
  sns.lineplot(data=df_frequencias, x='Palavras', y='Porcentagem', color='red', sort=False, ax=ax2)
  plt.show()

pareto(arquivo, "tratamento_mins", 10)

nuvem_palavras_s(arquivo, "tratamento_mins")
nuvem_palavras_neg(arquivo, "tratamento_mins")
