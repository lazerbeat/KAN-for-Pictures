import os
#import pickle
#import kan as kan
from imodelsx import KANClassifier


from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare data
input_dir = 'C:/Users/lazer/Desktop/KAN/clf-data'


categories = ['empty', 'not_empty']

data = []
labels = []
for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    #print(f"Checking directory: {category_path}")
    if not os.path.exists(category_path):
        #print(f"Directory does not exist: {category_path}")
        continue
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(category_path, file)
        #print(f"Processing file: {img_path}")
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
#classifier = SVC()

model = KANClassifier(device='cuda')

#training the model on training dataset (X_train, y_train)
model.fit(x_train, y_train)
#predicting output for the X_test dataset
y_pred = model.predict(x_test)


#accuracy on test dataset
accuracy_test = accuracy_score(y_test, y_pred)
print("\nAccuracy on Test Set:", accuracy_test)

#parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]


#grid_search = GridSearchCV(model, parameters)

#grid_search.fit(x_train, y_train)

# test performance
#best_estimator = grid_search.best_estimator_

#y_prediction = best_estimator.predict(x_test)

#score = accuracy_score(y_prediction, y_test)

#print('{}% of samples were correctly classified'.format(str(score * 100)))

#pickle.dump(best_estimator, open('./model.p', 'wb'))

