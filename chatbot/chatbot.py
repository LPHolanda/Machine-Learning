#!-*- coding: utf8 -*-
import numpy as np
import nltk
import random
from sklearn.model_selection import cross_val_score
from collections import Counter
import pandas as pd 

# ======= essa linha de download precisa ser executada caso seja a primeira vez na máquina para baixar a biblioteca =============
# nltk.download('stopwords')
# nltk.download('rslp')  ## para lidar com os sufixos das palavras. Exemplo: não colocar, carreira e carreiras na biblioteca.
# nltk.download("punkt") ## para lidar com as pontuações

texto1 = "quanto custa um plano premium?"
texto2 = "O exercício 15 do curso de Java 1 está com a resposta errada. Pode conferir por favor?"
texto3 = "Existe algum curso para cuidar do marketing da minha empresa?"

classificacoes = pd.read_csv('emails.csv', encoding='utf-8')
textosPuros = classificacoes['email']
frases = textosPuros.str.lower()
textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
stopwords = nltk.corpus.stopwords.words("portuguese")
stemmer = nltk.stem.RSLPStemmer()


# f=open('chatbot.txt','r',errors = 'ignore')
# raw=f.read()
# raw=raw.lower()# converts to lowercase
# # nltk.download('punkt') # first-time use only
# # nltk.download('wordnet') # first-time use only
# frase_tokens = nltk.sent_tokenize(frases)# converts to list of sentences 
# palavra_tokens = nltk.word_tokenize(frases)# converts to list of words

# sent_tokens[:2]
# word_tokens[:5]





dicionario = set()
for lista in textosQuebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)

 
totalDePalavras = len(dicionario)
print("Total de palavras no dicionário:", totalDePalavras)

# junta cada palavra a um numero chave incremental
tuplas = zip(dicionario, range(totalDePalavras))
tradutor = {palavra:indice for palavra,indice in tuplas}


def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor)

    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1

    return vetor

vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]
marcas = classificacoes['classificacao']

X = np.array(vetoresDeTexto)
Y = np.array(marcas)

porcentagem_de_treino = 0.8

tamanho_do_treino = int(porcentagem_de_treino * len(Y))
tamanho_de_validacao = len(Y) - tamanho_do_treino

treino_dados = X[0:tamanho_do_treino]
treino_marcacoes = Y[0:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]


def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv= k)
    taxa_de_acerto = np.mean(scores)

    print("Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto))
    return taxa_de_acerto


def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    acertos = (resultado == validacao_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    print("Taxa de acerto do vencedor no mundo real: {0}".format(taxa_de_acerto))


resultados = {}

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial 

maximo = max(resultados)
vencedor = resultados[maximo]
vencedor.fit(treino_dados, treino_marcacoes)

teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)
print("Total de testes: %d" % len(validacao_dados))



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

frase = texto2.lower()
textoQuebrado = nltk.word_tokenize(frase)

valida = [stemmer.stem(lista) for lista in textoQuebrado if lista not in stopwords and len(lista) > 2]
print(valida)

vetorDoTexto = [vetorizar_texto(valida, tradutor)]
print(dicionario)
print(vetorDoTexto)


modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial 

maximo = max(resultados)
vencedor = resultados[maximo]
vencedor.fit(treino_dados, treino_marcacoes)
 
teste_real(vencedor, vetorDoTexto, validacao_marcacoes)


# resultadoTexto = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, vetorDoTexto)
# print(resultadoTexto)


# print(len(dicionario))
# print(len(vetorDeTexto))
# print(frase)
# print(textoQuebrado)






















# GREETING_INPUTS = ("oi", "oie", "eae", "eai", "tudo bem",)
# GREETING_RESPONSES = ["oi", "oie", "eae", "eai", "tudo bem"]
# def greeting(sentence):
 
#     for word in sentence.split():
#         if word.lower() in GREETING_INPUTS:
#             return random.choice(GREETING_RESPONSES)  

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# def response(user_response):
#     robo_response=''



#     user_response = user_response.lower()
#     textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
#     vetoresDeTexto = [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]


#     resultado = modeloMultinomial.predict(vetoresDeTexto)

#     # X.append(user_response)
#     # TfidfVec = TfidfVectorizer(stopwords) ## formatação do texto
#     # tfidf = TfidfVec.fit_transform(X)
#     # vals = cosine_similarity(tfidf[-1], tfidf)
#     # idx=vals.argsort()[0][-2]
#     # flat = vals.flatten()
#     # flat.sort()
#     # req_tfidf = flat[-2]
#     # if(req_tfidf==0):
#     if(resultado == validacao_marcacoes):
#         robo_response=robo_response+"Perdão, eu faltei nessa aula"
#         return robo_response
#     else:
#         robo_response = robo_response+resultado
#         return robo_response



# flag=True
# print("Alura: Olá, o que gostaria de saber?")
# while(flag==True):
#     user_response = input()
#     user_response=user_response.lower()
#     if(user_response!='tchau'):
#         if(user_response=='obrigado' or user_response=='valeu' ):
#             flag=False
#             print("Alura: Até a próxima...")
#         else:
#             if(greeting(user_response)!=None):
#                 print("Alura: "+greeting(user_response))
#             else:
#                 print("Alura: ",end="")
#                 print(response(user_response))
#                 X.remove(user_response)
#     else:
#         flag=False
#         print("Alura: Até a próxima...")