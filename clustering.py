import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

def cluster_graph(directed_adj_matrix: np.array, theta: float, k: int, dictionary_form: bool = True) -> dict:
	n = len(directed_adj_matrix)
	clusters = defaultdict(lambda: defaultdict(int))
	if k <= 0:
	  return
	for m in range(1,k+1):
	  matrix = np.linalg.matrix_power(directed_adj_matrix, m)
	  for i in range(n):
	    for j in range(n):
	      if matrix[i][j] > theta:
	        clusters[i][j] = matrix[i][j]
	        print(clusters)
	for i in clusters.keys():
         clusters[i] = dict(clusters[i])
	clusters = dict(clusters)
	if not dictionary_form:
	  for i in range(n):
           clusters[i] = clusters[i].keys()
	return clusters

def dict_to_tensor(clusters: dict) -> torch.Tensor:
	n = len(clusters.keys())
	tensor = torch.zeros([n, n], dtype = torch.float32)
	for i in range(n):
	  if i not in clusters.keys():
	    tensor[i] = torch.ones([n], dtype = torch.float32)
	  else:
	    cluster = clusters[i]
	    for j in range(n):
	      if j not in cluster.keys():
	        tensor[i][j] = 0
	      else: 
	        tensor[i][j] = cluster[j]
	return tensor

def class_cluster(input: np.array, n_classes: int, theta: float, k: int) -> torch.Tensor:
	tensor = dict_to_tensor(cluster_graph(input, theta, k, True))
	counts = torch.count_nonzero(tensor, dim = 1)
	indices = tensor.topk(n_classes).indices
	tensor = tensor[indices]
	for i in range(tensor.size()[1]):
	  tensor[:,i] = torch.nn.functional.one_hot(torch.argmax(tensor[:, i]), tensor.size()[0])
	return tensor  

def plot_clusters(input: np.array, theta: float, k: int, n_classes: int, embeddings: torch.Tensor,  n_components: int = 2) -> None:
	z = class_cluster(input, n_classes, theta, k).numpy()
	tsne = TSNE(n_components)
	tsne_results = tsne.fit_transform(embeddings.numpy())
	if n_components == 2:
	  for i in n_clusters:
	    plt.scatter(tsne_results[z[0] == 1 , 0] , tsne_results[z[0] == 1 , 1] , label = i)
	    plt.scatter(np.mean(tsne_results[z[0] == 1 , 0]) , np.mean(tsne_results[z[0] == 1 , 1]) , s = 80, color = 'k')
	elif n_components == 3:
          fig = plt.figure(figsize=(12, 12))
	  ax = fig.add_subplot(projection='3d')
	  for i in n_clusters:
            ax.scatter(tsne_results[z[0] == 1 , 0] , tsne_results[z[0] == 1 , 1] , tsne_results[z[0] == 1 , 2] , label = i)
            ax.scatter(np.mean(tsne_results[z[0] == 1 , 0]) , np.mean(tsne_results[z[0] == 1 , 1]) , np.mean(tsne_results[z[0] == 1 , 2]) ,  s = 80, color = 'k')
	plt.legend()
	plt.show()  

