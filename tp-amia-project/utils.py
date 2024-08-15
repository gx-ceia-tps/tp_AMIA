import numpy as np
from numpy.linalg import det, inv


class ClassEncoder:
    def fit(self, y):
        self.names = np.unique(y)
        self.name_to_class = {name: idx for idx, name in enumerate(self.names)}
        self.fmt = y.dtype
        # Q1: por que no hace falta definir un class_to_name para el mapeo inverso?

    def _map_reshape(self, f, arr):
        return np.array([f(elem) for elem in arr.flatten()]).reshape(arr.shape)
        # Q2: por que hace falta un reshape?

    def transform(self, y):
        return self._map_reshape(lambda name: self.name_to_class[name], y)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def detransform(self, y_hat):
        return self._map_reshape(lambda idx: self.names[idx], y_hat)


class BaseBayesianClassifier:
    def __init__(self):
        self.encoder = ClassEncoder()

    def _estimate_a_priori(self, y):
        a_priori = np.bincount(y.flatten().astype(int)) / y.size
        # Q3: para que sirve bincount?
        return np.log(a_priori)

    def _fit_params(self, X, y):
        # estimate all needed parameters for given model
        raise NotImplementedError()

    def _predict_log_conditional(self, x, class_idx):
        # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
        # this should depend on the model used
        raise NotImplementedError()

    def fit(self, X, y, a_priori=None):
        # first encode the classes
        y = self.encoder.fit_transform(y)

        # if it's needed, estimate a priori probabilities
        self.log_a_priori = self._estimate_a_priori(y) if a_priori is None else np.log(a_priori)

        # check that a_priori has the correct number of classes
        assert len(self.log_a_priori) == len(
            self.encoder.names), "A priori probabilities do not match number of classes"

        # now that everything else is in place, estimate all needed parameters for given model
        self._fit_params(X, y)
        # Q4: por que el _fit_params va al final? no se puede mover a, por ejemplo, antes de la priori?

    def predict(self, X):
        # this is actually an individual prediction encased in a for-loop
        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=self.encoder.fmt)

        for i in range(m_obs):
            encoded_y_hat_i = self._predict_one(X[:, i].reshape(-1, 1))
            y_hat[i] = self.encoder.names[encoded_y_hat_i]

        # return prediction as a row vector (matching y)
        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        # calculate all log posteriori probabilities (actually, +C)
        log_posteriori = [log_a_priori_i + self._predict_log_conditional(x, idx) for idx, log_a_priori_i
                          in enumerate(self.log_a_priori)]

        # return the class that has maximum a posteriori probability
        return np.argmax(log_posteriori)


class QDA(BaseBayesianClassifier):

    def _fit_params(self, X, y):
        # estimate each covariance matrix
        # columnas son las muestras, filas te da la feature
        self.inv_covs = [inv(np.cov(X[:, y.flatten() == idx], bias=True))
                         for idx in range(len(self.log_a_priori))]
        # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
        # Q6: por que se usa bias=True en vez del default bias=False?
        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]
        # Q7: que hace axis=1? por que no axis=0?

    def _predict_log_conditional(self, x, class_idx):
        # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
        # this should depend on the model used
        inv_cov = self.inv_covs[class_idx]
        unbiased_x = x - self.means[class_idx]
        return 0.5 * np.log(det(inv_cov)) - 0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X, y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_x
#    -0.5 -> bug en codigo provisto por consigna
        return -0.5 * np.log(det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

    def _predict_one_custom(self, x):
        # return the class that has maximum a posteriori probability
        print(self._predict_log_conditionals(x))
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))

    def _predict_custom(self, X):
        # this is actually an individual prediction encased in a for-loop
        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=self.encoder.fmt)

        for i in range(m_obs):
            encoded_y_hat_i = self._predict_one_custom(X[:, i].reshape(-1, 1))
            y_hat[i] = self.encoder.names[encoded_y_hat_i]

        # return prediction as a row vector (matching y)
        return y_hat.reshape(1, -1)


# hiperparámetros


# preparing data, train - test validation
# 70-30 split
from sklearn.model_selection import train_test_split


def split_transpose(X, y, test_sz, random_state):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_sz, random_state=random_state)

    # transpose so observations are column vectors
    return X_train.T, y_train.T, X_test.T, y_test.T


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

# rng_seed = 6543
# train_x, train_y, test_x, test_y = split_transpose(X_full, y_full, 0.4, rng_seed)
