'''
loss functions for regression and classification 
'''

import numpy as np

class Metrics:

    def mean_squared_error(self,y_pred,y_actual):

        return np.mean((y_pred - y_actual)**2)
    
    
    def mean_absolute_error(self,y_pred,y_actual):
        
        return np.mean(np.abs((y_pred - y_actual)))
    
    
    def r2_score(self , y_pred,y_actual):


        rss =  np.sum((y_pred - y_actual)**2)
        mss  = np.sum((y_actual - np.mean(y_actual) )**2)

        return 1 - (rss/mss)
    

    def precision(self,y_pred,y_actual):

        true_positives = np.sum((y_pred == 1) &  (y_actual==1))       #when predicted =1 and actual=1      
        false_positives = np.sum((y_pred == 1) &  (y_actual==0))   #when predicted =1 and actual=0
        return true_positives / (true_positives + false_positives)
    
    def recall(self,y_pred,y_actual): 

        true_positives = np.sum((y_pred == 1) &  (y_actual==1)) 
        false_negative = np.sum((y_pred == 0) &  (y_actual==1))
        return true_positives / (true_positives + false_negative)

    def confusion_matrix(self,y_pred,y_actual):
            tp = np.sum((y_pred == 1) & (y_actual == 1))
            tn = np.sum((y_pred == 0) & (y_actual == 0))
            fp = np.sum((y_pred == 1) & (y_actual == 0))
            fn = np.sum((y_pred == 0) & (y_actual == 1))

            return np.array([[tn, fp],
                            [fn, tp]])
            
    def accuracy(self,y_pred,y_actual):
        true_positives = np.sum((y_pred == 1) &  (y_actual==1))
        false_positives = np.sum((y_pred == 1) &  (y_actual==0))
        true_positives = np.sum((y_pred == 1) &  (y_actual==1))
        false_negative = np.sum((y_pred == 0) &  (y_actual==1))
        true_negative = np.sum((y_pred == 0) &  (y_actual==0))


        return (true_positives + true_negative)/ (true_positives + true_negative + false_positives + false_negative)
        
    def f1_score(self,y_pred,y_actual):
        true_positives = np.sum((y_pred == 1) &  (y_actual==1))
        false_positives = np.sum((y_pred == 1) &  (y_actual==0))
        true_positives = np.sum((y_pred == 1) &  (y_actual==1))
        false_negative = np.sum((y_pred == 0) &  (y_actual==1))
        true_negative = np.sum((y_pred == 0) &  (y_actual==0))

        precision=  true_positives / (true_positives + false_positives)
        recall= true_positives / (true_positives + false_positives)

        return 2 * precision * recall / (precision + recall)  
    


    

# def main():
#     y_actual = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
#     y_pred   = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0])

#     m = Metrics()

#     print("=== Regression Metrics ===")
#     print("MSE :", m.mean_squared_error(y_pred, y_actual))
#     print("MAE :", m.mean_absolute_error(y_pred, y_actual))
#     print("R2  :", m.r2_score(y_pred, y_actual))

#     print("\n=== Classification Metrics ===")
#     print("Accuracy :", m.accuracy(y_pred, y_actual))
#     print("Precision:", m.precision(y_pred, y_actual))
#     print("Recall   :", m.recall(y_pred, y_actual))
#     print("F1 Score :", m.f1_score(y_pred, y_actual))
#     print("Confusion Matrix:\n", m.confusion_matrix(y_pred, y_actual))


# if __name__ == "__main__":
#     main()