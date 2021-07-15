from sklearn.datasets import load_digits
digits = load_digits()

''' Create the TSNE object; t-distributed Stochastic Neighbor Embedding (t-SNE)'''
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=11)

reduced_data = tsne.fit_transform(digits.data)

# visualize the reduced data set 
import matplotlib.pyplot as pyplot
dots = pyplot.scatter(reduced_data[:, 0], reduced_data[:,1], c='black')
pyplot.show()

dots = pyplot.scatter(reduced_data[:, 0], reduced_data[:,1], c=digits.target, cmap=pyplot.cm.get_cmap('nipy_spectral_r', 10))

cmap = pyplot.cm.get_cmap('nipy_spectral_r', 10)
