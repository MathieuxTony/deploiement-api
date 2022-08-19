import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


data = pd.read_csv('data.csv')[['Tags', 'title_bow_lem']]

for c in ['Tags', 'title_bow_lem']:
      data[c] = data[c].transform(lambda x: eval(x))

X = data['title_bow_lem'].apply(lambda x: ' '.join(x))
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(data['Tags'])
y = pd.DataFrame(y, columns=multilabel.classes_)

df_labels = pd.DataFrame(columns=multilabel.classes_)
df_labels.to_csv('label_pred.csv', index=False)

# grille d'hyperparamètres
param_grid = {'C': [0.5, 1, 2, 5]}

# initialiser la validation croisée
grid_pred = GridSearchCV(
        LinearSVC(),
        param_grid,
        cv=5)

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', OneVsRestClassifier(grid_pred, n_jobs=1)),
            ])


pipeline.fit(X, y)

dump(pipeline, './mon_model.joblib')