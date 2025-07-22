# About PCA

# PCA provides a roadmap for how to reduce a complex data set to a lower dimension to reveal the
# sometimes hidden, simplified structures that often underlie it.




# Steps to calculate pca 

# 1- Mean centre the data.
# 2- Find the covariance matrix.
# 3- Find the eigenvalue and eigenvectors of the covariance matrix.
# 4- Then find the eigen vector who has highest eigen value(as this is the one,  which has max variance , it will be principal component 1)
# 5- We just need to project all our points on this using dot product between our features and the prinipal component.

import numpy as np

class PCA:

    def __init__(self , n_components:int = None):

        '''
        This will initialize the PCA class ,with n as the number 
        of principal components we want. 
        '''

        self.n_components = n_components
        

    def fit_transform(self , x_train):
        
        #1. Standardise the data

        x_train_mean  = np.mean(x_train,axis=0)
        x_train_standardised = x_train  - x_train_mean


        #2. covariance matrix

        covar_matrix = np.cov(x_train_standardised , rowvar=False)

        #3. find the eigen stuff of covar_matrix and first n_components

        eigen_values , eigen_vectors  = np.linalg.eigh(covar_matrix)

        #4. sort from highest values of eigen values and extract top n_components

        sorted_idx = np.argsort(eigen_values)[::-1]
        eigenvalues = eigen_values[sorted_idx]
        eigenvectors = eigen_vectors[:, sorted_idx]
        
        #5. dot product between our features and top n eigen vectors

        top_n_eigen_vectors  = eigenvectors[ : , :self.n_components]

        x_pca = x_train_standardised  @  top_n_eigen_vectors

        self.mean = x_train_mean
        self.covariance = covar_matrix
        self.top_n_eigen_vectors = top_n_eigen_vectors

        return x_pca




    def transform(self , x_test):
        '''
        In fit we will use the learnt statistics about our data 
        basically we transform the test data by projecting to learnt eigenvectors 
        
        '''

        x_centered = x_test - self.mean
  
        x = x_centered @ self.top_n_eigen_vectors
        
        return x
  

  


# if __name__ == "__main__":
#     pca = PCA(1)
#     X = np.array([
#   [2, 3, 4],
#   [4, 5, 6],
#   [6, 7, 8],
#   [10, 2, 3]
# ])
    
#     y  = np.array([  [6, 7, 8],
#             [10, 2, 3]])    

#     jj = pca.fit_transform(X)
#     ot = pca.transform(y)
#     print(ot)


#     print("---------now sklearns---------")

#     from sklearn.decomposition import PCA

#     OBJ = PCA(1)

#     jj = OBJ.fit_transform(X)
#     ot = OBJ.transform(y)
#     print(ot)



