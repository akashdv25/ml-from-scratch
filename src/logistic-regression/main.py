'''
Turn of events 

1- Perceptron trick - 

we want a line which can correctly classify two labels , we go to each point and
ask wheter it is correctly clasiified yes or no if yes then we do nothing 
weight new = weight old , bias new = bias old (y=mx +b for deciding eq of line)
but if point is incorrect we just push the line and shift it by using this

weight_new = weight_old + n * (y_actual - y_pred(label)* input_feature)

but this will give us a linear classifier which is not the best we cant quantify this.

'''


'''
2- Introducing sigmoid into the mix

sigmoid takes any value and squishes it between 0 and 1 (so that can be expressed in 
terms of prob)

y_pred = sigmoid(linear combination of weights and bias)

weight_new = weight_old + n * (y_actual - y_pred(probability)* input_feature)

now this will update the weight and we will also use the magintude of push
if point is correctly classiifed and closer to the line the line will be
pushed away with greater force as compared to when point was correctly classified and 
far away from the line.

So we will get a better fit.

'''


#maximum likelihood means we want that classifier line which has highest correct predicted
#probabilities for labels i.e highest mle (we multiply probabilties p(y=1|x))

#prblm in mle = product (small number = very small number) so convert to sum

# side note why not just take a linear classifier like perceptron

# Becasue we wont be able to quantify it and tell which classifier line is better 
# Whereas we can classify while working with a loss function(sigmoid and probabilities)
 
'''
3- Introducing the loss function 

Loss func is neeeded to compare two classifier and also get the best params.

mle ka log = negative log loss , negative due to log(0-1 = is always -ve) [maximize]

mle = p(y=1|x) * p(y=0|x)*....

but when we calcuate log loss we need to minimzie because log(0.1) > log(0.7) and log * = addition


loss = (-(y_actual(label)log(y_pred(prob)) - 1-y_actual(label)log(1-y_pred(prob)) 



'''

'''
4- Minimizing the loss func and getting best weights train for some epochs
'''

import numpy as np

class LogisticRegression:

    def __init__(self, epochs=100, n=0.001):
        self.weight = None
        self.bias = None
        self.epochs = epochs
        self.n = n

    def _sigmoid(self, z):
        #  sigmoid
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def fit(self, x_train, y_train):
        n_samples, n_features = x_train.shape
        self.weight = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        for i in range(self.epochs):
            z = np.dot(x_train, self.weight) + self.bias
            y_pred = self._sigmoid(z)

            # Clip predictions to avoid log(0)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            loss = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))

            # Gradients
            dw = np.dot((y_pred - y_train), x_train) / n_samples
            db = np.sum(y_pred - y_train) / n_samples

            # Update parameters
            self.weight -= self.n * dw
            self.bias -= self.n * db

            if i % 10 == 0:
                print(f"Epoch {i} | Loss: {loss:.4f}")

        return self.weight, self.bias

    def predict_proba(self, x_test):
        z = np.dot(x_test, self.weight) + self.bias
        return self._sigmoid(z)

    def predict(self, x_test):
        proba = self.predict_proba(x_test)
        return (proba >= 0.5).astype(int)



# # --- Main Block ---
# if __name__ == "__main__":
#     x_train = np.array([[i/10] for i in range(5, 45, 2)])
#     y_train = np.array([0]*9 + [1]*11)

#     x_test = np.array([
#         [1.0],
#         [2.0],
#         [2.5],
#         [3.0],
#         [4.0]
#     ])

#     obj = LogisticRegression(epochs=100, n=0.1)
#     obj.fit(x_train, y_train)

#     print("\nPredicted classes on test:")
#     print(obj.predict(x_test))

#     print("\nPredicted probabilities on test:")
#     print(obj.predict_proba(x_test))
