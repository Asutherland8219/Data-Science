### Dimension reduction techniques such as principal component analysis (PCA) are use dt ohelp reduce the number of dimensions thus creating a better model. Models with less dimensions (2-3) are easier to be visualized. The idea behind PCA is to find the variables that contain the most information. ###
## Aka it creates a linear combination of 2 variables that contains the most information. It yiels the directions that maximize the variance of the data. ##
from sklearn import PCA

# Create a Randomized PCA model that takes 2 components #
randomized_pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_rpca = randomized_pca.fit_transform(digits.data)

# Creat a regular PCA model 
pca = PCA(n_components=2)

# Fit and transform the data to the model
reduced_data_pca = pca.fit_transform(digits.data)

