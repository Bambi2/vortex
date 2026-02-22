import numpy as np
import scipy
import sklearn

class CSP(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, n_components: int = 4):
        self.n_components = n_components

    def fit(self, x, y):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")

        class_1_data = x[y == classes[0]]
        class_2_data = x[y == classes[1]]

        class1_cov = self._compute_covariance_matrix(class_1_data)
        class2_cov = self._compute_covariance_matrix(class_2_data)
        total_cov = class1_cov + class2_cov

        eigenvalues, eigenvectors = scipy.linalg.eigh(class1_cov, total_cov)
        descending_indices = np.flip(np.argsort(eigenvalues))
        eigenvectors, eigenvalues = eigenvectors[:, descending_indices], eigenvalues[descending_indices]

        n = self.n_components // 2
        selected_indexes = list(range(n))+list(range(-n, 0))
        self.filters_ = eigenvectors[:, selected_indexes].T

        return self

    def transform(self, x):
        filtered_x = np.array([self.filters_ @ epoch for epoch in x])
        features = np.log(np.var(filtered_x, axis=2))

        return features

    def _compute_covariance_matrix(self, x):
        number_of_epochs, number_of_channels, number_of_time_series = x.shape
        covs = np.zeros([number_of_epochs, number_of_channels, number_of_channels])

        for i, epoch in enumerate(x):
            cov = epoch @ epoch.T / number_of_time_series
            covs[i] = cov / np.trace(cov)
        
        return np.mean(covs, axis=0)