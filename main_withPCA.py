import numpy as np
from sklearn.metrics import confusion_matrix


'''
Input:
x: An array or vector representing the coordinates of the first point.
y: An array or vector representing the coordinates of the second point.

Output:
The function returns a single value, which is the Euclidean distance between the two points x and y. The Euclidean distance is a measure of the straight-line distance between two points in Euclidean space. It is calculated as the square root of the sum of the squared differences between corresponding coordinates of the two points.
'''


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


'''
Input:
data: A 2-dimensional array or matrix representing the dataset. Each row corresponds to a sample, and each column corresponds to a feature.
k: An integer indicating the desired number of clusters to be formed.

Output:
The function returns an array cluster_labels of length n_samples, where n_samples is the number of samples in the dataset. Each element in cluster_labels represents the cluster assignment for the corresponding sample.
'''


def hierarchical_clustering(data, k):
    n_samples, n_features = data.shape

    # Initialize each sample as a cluster
    clusters = [{'id': i, 'center': data[i], 'members': [i]}
                for i in range(n_samples)]

    # Merge clusters until there are k clusters
    while len(clusters) > k:
        # Find the pair of clusters with the smallest distance between them
        min_distance = float('inf')
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                distance = euclidean_distance(
                    clusters[i]['center'], clusters[j]['center'])
                if distance < min_distance:
                    min_distance = distance
                    min_i, min_j = i, j

        # Merge the two clusters with the smallest distance
        merged_center = (clusters[min_i]['center'] +
                         clusters[min_j]['center']) / 2
        merged_members = clusters[min_i]['members'] + \
            clusters[min_j]['members']
        merged_cluster = {
            'id': len(clusters), 'center': merged_center, 'members': merged_members}
        del clusters[max(min_i, min_j)]
        del clusters[min(min_i, min_j)]
        clusters.append(merged_cluster)

    # Assign cluster IDs to each sample
    cluster_labels = np.zeros(n_samples)
    for i, cluster in enumerate(clusters):
        for member in cluster['members']:
            cluster_labels[member] = i

    return cluster_labels


'''
Input:
train_data: A 2-dimensional array or matrix representing the training dataset. Each row corresponds to a training sample, and each column corresponds to a feature.
train_labels: An array or list containing the labels corresponding to the training samples. The length of train_labels should be equal to the number of rows in train_data.
test_data: A 2-dimensional array or matrix representing the test dataset. Each row corresponds to a test sample, and each column corresponds to a feature.
k: An integer indicating the number of nearest neighbors to consider for classification.

Output:
The function returns an array predictions of length n_test, where n_test is the number of samples in the test dataset. Each element in predictions represents the predicted label for the corresponding test sample.
'''


def knn_classify(train_data, train_labels, test_data, k):
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    predictions = np.zeros(n_test)

    for i in range(n_test):
        distances = [euclidean_distance(
            test_data[i], train_data[j]) for j in range(n_train)]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        max_count_label = unique_labels[np.argmax(counts)]
        predictions[i] = max_count_label

    return predictions


'''
Input:
data: A 2-dimensional array or matrix representing the dataset. Each row corresponds to a sample, and each column corresponds to a feature.
n_components: An integer indicating the number of principal components (dimensions) to retain in the transformed data.

Output:
The function returns a new array transformed_data of shape (n_samples, n_components), where n_samples is the number of samples in the original dataset. 
'''


def pca(data, n_components):
    # Compute the covariance matrix
    cov_matrix = np.cov(data.T)

    # Perform eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Transform the data to the new subspace
    transformed_data = np.dot(data, selected_eigenvectors)

    return transformed_data


data = np.loadtxt('Seed_Data.csv', delimiter=',', skiprows=1)
data_features = data[:, :-1]
data_labels = data[:, -1]

# Normalize the features
data_features = (data_features - np.mean(data_features, axis=0)
                 ) / np.std(data_features, axis=0)

# Apply PCA to reduce dimensionality
n_components = 2

# Perform PCA on the data
transformed_data = pca(data_features, n_components)

# Perform hierarchical clustering on the transformed data to get cluster labels
cluster_labels = hierarchical_clustering(transformed_data, k=3)

# Split the data into training and test sets
n_samples = transformed_data.shape[0]
train_indices = np.random.choice(
    n_samples, size=int(0.8*n_samples), replace=False)
test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
train_data = transformed_data[train_indices]
train_labels = cluster_labels[train_indices]
test_data = transformed_data[test_indices]
test_labels = cluster_labels[test_indices]

# Perform K-nearest neighbor classification on the test set
predictions = knn_classify(train_data, train_labels, test_data, k=3)
predictions1 = knn_classify(train_data, train_labels, train_data, k=3)
print('Confusion Matrix - Train: \n',
      confusion_matrix(train_labels, predictions1))
accuracy1 = np.mean(predictions1 == train_labels)
print('Accuracy for Training data: ', accuracy1)
print('Confusion Matrix - Test: \n', confusion_matrix(test_labels, predictions))
# Compute the accuracy of the classifier
accuracy = np.mean(predictions == test_labels)
print('Accuracy for Testing data: ', accuracy)
