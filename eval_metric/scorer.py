from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import glob

REAL_DATA_PATTERN = "data_*/*.npy"

def compare_repertoires(real, fake, test_size=0.5):

    # standardize  data
    # real = (real - real.mean(axis=0)) / (real.std(axis=0))
    # fake = (fake - fake.mean(axis=0)) / (fake.std(axis=0))

    data_X = np.concatenate([real, fake])
    data_Y = np.concatenate([np.ones((real.shape[0],)), np.zeros((fake.shape[0],))])

    data_X = (data_X - data_X.min(axis=0)) / (data_X.max(axis=0) - data_X.min(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=test_size, random_state=42)

    classifier = SVC(kernel='poly', gamma='scale')
    classifier.fit(X_train, y_train)

    return (classifier.score(X_train, y_train), classifier.score(X_test, y_test))

# Real songs
real_songs = np.concatenate([np.load(file_) for file_ in glob.glob(REAL_DATA_PATTERN)])

datasets = [
("Dataset", real_songs),
("Dataset + N(0, 0.1)", real_songs + np.random.normal(loc=0, scale=0.1, size=real_songs.shape)),
("Dataset + N(0, 1)", real_songs + np.random.normal(loc=0, scale=1, size=real_songs.shape)),
("Dataset + N(0, 10)", real_songs + np.random.normal(loc=0, scale=10, size=real_songs.shape)),
("N(mu, sigma)", np.random.normal(loc=np.mean(real_songs, axis=0), scale=np.std(real_songs, axis=0), size=real_songs.shape)),
("Exp(sigma)", np.random.exponential(scale=np.std(real_songs, axis=0), size=real_songs.shape)),
("Log(mu, sigma)", np.random.logistic(loc=np.mean(real_songs, axis=0), scale=np.std(real_songs, axis=0), size=real_songs.shape)),
("Zero()", np.zeros(real_songs.shape)),
]

num_iters = 5
for key, fake_songs in datasets:
    accuracies = np.zeros(2)
    for _ in range(num_iters):
        accuracies += np.array(compare_repertoires(real_songs, fake_songs))
    accuracies /= num_iters
    print(key, accuracies)
