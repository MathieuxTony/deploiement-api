
# Déploiement d'une api avec streamlit

Le but est de déployer un modèle basique de classification supervisée de données textuelles.
Les textes considérés sont des questions posées sur Stack Overflow et l'idée sous-jacente est 
de suggérer des tags à la personne qui pose une question afin de la classer.

Il s'agit d'une classification multilabels où on se limite à seulement 50 labels (tags) différents, 
sur des données en bag of words sans sentence embedding afin que le modèle soit suffisamment 
léger pour Heroku.

Le modèle de classification utilisera un algorithme Linear SVC, un OneVsRestClassifier et 
un GridSearchCV afin d'optimiser le paramètre 'C' du linear SVC.


