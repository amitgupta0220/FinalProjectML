import matplotlib.pyplot as plt
import numpy as np

raw_data = np.loadtxt('Seed_Data.csv', skiprows=1, delimiter=',')


def euclidean_distance(x, y):
    distance = np.sqrt(np.sum((x - y)**2))
    # print(f'distance: {distance}, x: {x}, y: {y}')
    return distance


def hierarchical_clustering(data, n_clusters):
    n_samples, n_features = data.shape

    # Initialize each sample as a cluster
    clusters = [{'id': i, 'center': data[i], 'members': [i]}
                for i in range(n_samples)]

    # Merge clusters until the desired number of clusters is reached
    while len(clusters) > n_clusters:
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

    # Assign each sample to the closest cluster
    cluster_labels = np.zeros(n_samples)
    for i in range(len(clusters)):
        for j in clusters[i]['members']:
            cluster_labels[j] = i

    return cluster_labels.astype(int)


data = np.loadtxt('Seed_Data.csv', skiprows=1, delimiter=',')
data_normalized = (data[:, :-1] - np.mean(data[:, :-1],
                   axis=0)) / np.std(data[:, :-1], axis=0)

# Determine the optimal number of clusters using the elbow method
ssd = []
for k in range(1, 11):
    cluster_labels = hierarchical_clustering(data_normalized, k)
    centers = [np.mean(data_normalized[cluster_labels == i], axis=0)
               for i in range(k)]
    ssd.append(np.sum([(data_normalized[i] - centers[cluster_labels[i]])
               ** 2 for i in range(len(data_normalized))]))

plt.plot(range(1, 11), ssd, 'o-')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster (sum of squared distances)')
plt.title('Elbow Method')
plt.show()
