import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class datamodel():
   def __init__(self):
      version="1.0"
   def generate_sample_data(self, split_size):
      np.random.seed(2)
      
      train_size=split_size
      x = np.linspace(0,30,100)
      np.random.shuffle(x)
      
      Xtrain = x[:train_size, None]
      Ytrain = x[train_size:, None]
      
      labels_train = np.sin(Xtrain,)
      labels_test = np.sin(Ytrain,)
      return  Xtrain, Ytrain, labels_train, labels_test

   def modelfit(self, model, Xtrain, labels_train, Ytrain):
      model.fit(Xtrain, labels_train)
      train_predictions = model.predict(Xtrain)
      predictions = model.predict(Ytrain)
      return predictions, train_predictions
      
   def print_crossval(self, model, Xtrain, labels_train):
      trainfit = cross_val_score(model, Xtrain, labels_train)
      print "Score = %s"%(trainfit)

   def plot_data(self, modelname, Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions):
      plt.title(modelname + " : Test performance")
      plt.scatter(Ytrain, labels_test, c = 'r')
      plt.scatter(Ytrain, predictions, c = 'b')
      plt.show()
      
      plt.title(modelname + " : Train performance")
      plt.scatter(Xtrain, labels_train, c = 'r')
      plt.scatter(Xtrain, train_predictions, c = 'b')
      plt.show()

if __name__ == '__main__':
   obj = datamodel()
   Xtrain, Ytrain, labels_train, labels_test = obj.generate_sample_data(80)

   #Model : Decision Tree
   model = DecisionTreeRegressor(max_depth=10)
   predictions, train_predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   obj.print_crossval(model, Xtrain, labels_train) 
   obj.plot_data("Decision Tree", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)
   
   #Model : Random Forest Regressor 
   model = RandomForestRegressor()
   predictions, train_predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   
   obj.print_crossval(model, Xtrain, labels_train) 
   obj.plot_data("Random Forest", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)

