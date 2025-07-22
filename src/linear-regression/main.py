
'''
So linear regression algo can be made  using ols(closed form solution)
or gradient descent to find the minima of loss function 

loss = mse . mape , mae , etc , even though in sklearn implementation we use 
ols 

'''

import numpy as np

class LinearRegression:
    
    def __init__(self ):
        pass



        

    def fit(self , x_train , y_train):

        '''
        y = m1x1 + m2x2 + m3x3 + .... + b

        coef_ & intercept vals we want such that our loss is minimum

        intercept = y bar - m x bar

                    summattion i=1 to n (xi - x bar) (yi - y bar)
        coef_ =   ----------------------------------------------
                       summattion i=1 to n (xi - x bar) ** 2


        for multiple linear regression we have
        
                               

        '''    

        # self.coeficient = sum(x_train - np.mean(x_train)) *(y_train - np.mean(y_train)) / sum(x_train - np.mean(x_train)** 2)
        # self.bias = np.mean(y_train) - self.coeficient * np.mean(x_train)

        # return self.coeficient , self.bias


        n_samples = x_train.shape[0]
        X_b = np.hstack([np.ones((n_samples, 1)), x_train])  # shape: (n_samples, n_features + 1)

        # Closed-form OLS
        theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train

        self.bias_ = theta_best[0]
        self.coef_ = theta_best[1:]
        return self.coef_, self.bias_
        


    def predict(self,x_test):

        '''Get y = mx+c'''

        return ( x_test @ self.coef_ ) + self.bias_


    def mse(self,y_actual,y_pred):

        return np.mean((y_actual - y_pred)**2)





# if __name__ == "__main__":
    
#     x_train = np.array([
#     [1.4, 2],   # 1400 sq ft, 2 bedrooms
#     [1.8, 3],   # 1800 sq ft, 3 bedrooms
#     [2.4, 4]    # 2400 sq ft, 4 bedrooms
# ])

#     y_train = np.array([
#     50,   # 50 lakhs
#     65,   # 65 lakhs
#     80    # 80 lakhs
# ])


#     x_test = np.array([
#     [2.0, 3],   # 2000 sq ft, 3 bedrooms
#     [1.6, 2]    # 1600 sq ft, 2 bedrooms
# ])

#     y_test = np.array([
#     70,   # Expected ~70 lakhs
#     55    # Expected ~55 lakhs
# ])


#     lr1 = LinearRegression()
#     lr1.fit(x_train,y_train)
#     y_pred=lr1.predict(x_test)
#     print(lr1.mse(y_test,y_pred))

#     print("=================now sk learn")

#     from sklearn.linear_model import LinearRegression
#     from sklearn.metrics import mean_squared_error
#     lr = LinearRegression()
#     lr.fit(x_train,y_train)
#     y_pred = lr.predict(x_test)
#     print(mean_squared_error(y_test,y_pred))




