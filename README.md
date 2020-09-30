# Modelos de Análise de Sentimento Utilizando Avaliações Online de Produtos e Serviços

Trabalho de Conclusão de Pós-graduação Lato Sensu em Ciência de Dados e Big Data na Pontifícia Universidade Católica De Minas Gerais.

Autor: Alexandre Henriques Nunes

## Instruções

O trabalho foi feito utilizando Python versão 3.7.5 64-bits. As dependências precisam ser instaladas antes de executar os scripts.

Existem dois projetos neste repositório, sendo o primeiro o scraper, responsável por extrair os dados necessários para o modeo. O segundo projeto é o model de aprendizado de máquina.

Pra rodar o scraper o seguinte comando pode ser utilizado dentro do diretório "scraper":

```
scrapy list | xargs -P 0 -n 1 scrapy crawl
```

Para rodar o model primeiramente o arquivo "data.zip" dentro do diretório "model/data" precisa ser descompactado e a seguir pode ser utilizado o seguinte comando dentro do diretório "model":

```
python main.py
```

## Dependências

```
# Framework for extracting from websites - https://scrapy.org/
pip install Scrapy

# Lightweight pipelining - https://joblib.readthedocs.io/
pip install joblib

# Plotting - https://matplotlib.org/
pip install matplotlib

# Natural language processing - http://nltk.org/
pip install nltk

# Array computing - https://www.numpy.org/
pip install numpy

# Expressive data structures - https://pandas.pydata.org/
pip install pandas

# Machine learning - http://scikit-learn.org/
pip install scikit-learn

# Natural language processing - https://spacy.io/
pip install spacy

# Unicode text handler - https://pypi.org/project/Unidecode/
pip install unidecode

# Fast language prediction - https://github.com/indix/whatthelang
pip install whatthelang

# Word cloud generator - https://github.com/amueller/word_cloud
pip install wordcloud

# Spell checker - https://github.com/mammothb/symspellpy
pip install symspellpy

# Gradient Boosting - https://xgboost.ai/
pip install xgboost

# Gradient Boosting - https://github.com/microsoft/LightGBM
pip install lightgbm

# Gradient Boosting - https://catboost.ai/
pip install catboost

# Deep Learning - https://www.tensorflow.org/
pip install tensorflow

# A set of utility functions for iterators, functions and dictionaries - https://github.com/pytoolz/toolz/
pip install toolz

# Portuguese data for Spacy - https://spacy.io/
python -m spacy download pt
```
