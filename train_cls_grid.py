import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

data_dict = pickle.load(open("data_ada_shabrinya.pickle", 'rb'))

group_21 = {
    'data':[],
    'labels':[]
}

group_42 = {
    'data':[],
    'labels':[]
}

for label, data in zip(data_dict['labels'], data_dict['data']):

    if len(data) == 21:
        group_21['data'].append(data)
        group_21['labels'].append(label)
    
    else:
        group_42['data'].append(data)
        group_42['labels'].append(label)

data_21 = np.asarray(group_21['data'])
data_21_2dim = np.asarray([data.flatten() for data in data_21])
labels_21 = np.asarray(group_21['labels'])

data_42 = np.asarray(group_42['data'])
data_42_2dim = np.asarray([data.flatten() for data in data_42])
labels_42 = np.asarray(group_42['labels'])

#21 Group

X_21 = np.array(data_21_2dim)
y_21 = np.array(labels_21)

X_train, X_test, y_train, y_test = train_test_split(X_21, y_21, test_size=0.2, shuffle=True, stratify=y_21, random_state=42)

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],  # Jumlah pohon
    'max_depth': [10, 20, 30, None],  # Kedalaman maksimum pohon
    'min_samples_split': [2, 5, 10],  # Jumlah minimum sampel untuk split
    'min_samples_leaf': [1, 2, 4],    # Jumlah minimum sampel di leaf
    'max_features': ['auto', 'sqrt']  # Fitur yang dipertimbang
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print(f'Akurasi model 21_group: {accuracy_score(y_test, y_pred)*100}%')

f = open("best_model21.p", 'wb')
pickle.dump({'model':model}, f)
f.close()

#42 Group

X_42 = np.array(data_42_2dim)
y_42 = np.array(labels_42)

X_train, X_test, y_train, y_test = train_test_split(X_42, y_42, test_size=0.2, shuffle=True, stratify=labels_42, random_state=42)

model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],  # Jumlah pohon
    'max_depth': [10, 20, 30, None],  # Kedalaman maksimum pohon
    'min_samples_split': [2, 5, 10],  # Jumlah minimum sampel untuk split
    'min_samples_leaf': [1, 2, 4],    # Jumlah minimum sampel di leaf
    'max_features': ['auto', 'sqrt']  # Fitur yang dipertimbang
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_y_pred = model.predict(X_test)

print(f'Akurasi model 42_group: {accuracy_score(y_test, y_pred)*100}%')

f = open("best_model42.p", 'wb')
pickle.dump({'model':best_model}, f)
f.close()

