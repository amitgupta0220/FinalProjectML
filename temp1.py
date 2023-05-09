import matplotlib.pyplot as plt
import numpy as np

# Load data
data = np.loadtxt('Seed_Data.csv', delimiter=',', skiprows=1)

# Calculate Euclidean distance between two points


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Calculate similarity between two clusters using the average linkage method


def average_linkage_similarity(c1, c2):
    similarity = 0
    for i in range(len(c1)):
        for j in range(len(c2)):
            similarity += euclidean_distance(c1[i], c2[j])
    return similarity / (len(c1) * len(c2))

# Perform hierarchical clustering using the average linkage method


def hierarchical_clustering(data, k):
    # Initialize clusters with each data point as its own cluster
    clusters = []
    for i in range(len(data)):
        clusters.append([data[i]])

    # Merge clusters until there are k clusters remaining
    while len(clusters) > k:
        # Find two clusters with highest similarity
        max_similarity = -np.inf
        max_i = -1
        max_j = -1
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                similarity = average_linkage_similarity(
                    clusters[i], clusters[j])
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_i = i
                    max_j = j

        # Merge clusters
        new_cluster = clusters[max_i] + clusters[max_j]
        clusters.pop(max_j)
        clusters[max_i] = new_cluster

    return clusters

# Use the elbow method to determine the optimal number of clusters


def elbow_method(data, k_max):
    # Initialize list of SSE values
    sse_values = []

    # Iterate over number of clusters
    for k in range(1, k_max+1):
        # Perform hierarchical clustering
        clusters = hierarchical_clustering(data, k)

        # Calculate SSE for each cluster
        sse = 0
        for i in range(len(clusters)):
            cluster_mean = np.mean(clusters[i], axis=0)
            for j in range(len(clusters[i])):
                sse += euclidean_distance(clusters[i][j], cluster_mean)**2

        sse_values.append(sse)

    # Calculate differences between SSE values
    sse_diff = np.diff(sse_values)

    # Find optimal number of clusters
    optimal_k = 1
    max_diff = -np.inf
    for i in range(len(sse_diff)):
        if sse_diff[i] > max_diff:
            optimal_k = i+2
            max_diff = sse_diff[i]

    return optimal_k, sse_values

# Use the optimal number of clusters to classify the data using k-nearest neighbors


def knn_classification(data, labels, k_max, optimal_k):
    # Initialize list of classification accuracies
    accuracies = []

    # Iterate over number of neighbors
    for k in range(1, k_max+1):
        # Initialize list of predicted labels
        predicted_labels = []

        # Iterate over data points
        for i in range(len(data)):
            # Find k nearest neighbors
            distances = []
            for j in range(len(data)):
                if i == j:
                    continue
                distances.append((j, euclidean_distance(data[i], data[j])))
            distances = sorted(distances, key=lambda x: x[1])
            neighbors = [labels[x[0]] for x in distances[:k]]

            # Predict label
            cluster_ids = [np.where(np.array(list(data[j])) == np.array(
                list(clusters[i]))) for i in range(optimal_k) for j in range(len(data))]
            cluster_ids = np.array(cluster_ids).reshape(len(data), optimal_k)
            cluster_votes = [np.count_nonzero(
                cluster_ids[i] == j) for j in range(optimal_k)]
            predicted_label = np.argmax(cluster_votes)
            predicted_labels.append(predicted_label)

        # Calculate classification accuracy
        accuracy = np.mean(predicted_labels == labels)
        accuracies.append(accuracy)

    return accuracies


# Use the elbow method to determine the optimal number of clusters
k_max = 5
optimal_k, sse_values = elbow_method(data, k_max)

# Perform hierarchical clustering using the optimal number of clusters
clusters = hierarchical_clustering(data, optimal_k)

# Create list of cluster IDs for each data point
cluster_ids = []
for i in range(len(data)):
    for j in range(len(clusters)):
        if list(data[i]) in clusters[j]:
            cluster_ids.append(j)
            break

# Classify data using k-nearest neighbors
labels = data[:, -1].astype(int)
accuracy = knn_classification(data[:, :-1], labels, k_max, optimal_k)

print(f'Optimal number of clusters: {optimal_k}')
print(f'Classification accuracy using k-nearest neighbors: {accuracy:.3f}')

# Plot SSE vs. number of clusters
plt.plot(range(1, k_max+1), sse_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Plot data points with cluster IDs as colors
plt.scatter(data[:, 0], data[:, 1], c=cluster_ids)
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title(f'Hierarchical Clustering with {optimal_k} Clusters')
plt.show()
