import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
'''
Script fits 4 models :
    1. Single decision Tree
    2. Random Forest
    3. Bagged decision Tree
    4. Bagged Random Forest : Takes extra time
'''


class datamodel():
   def __init__(self):
      version="1.0"
   def generate_sample_data(self, split_size, N):
      np.random.seed(2)
      
      train_size=split_size
      x = np.linspace(0,30,N)
      np.random.shuffle(x)
      
      Xtrain = x[:train_size, None]
      Ytrain = x[train_size:, None]
      
      labels_train = np.sin(Xtrain,)
      labels_test = np.sin(Ytrain,)
      return  Xtrain, Ytrain, labels_train, labels_test

   def modelfit(self, model, Xtrain, labels_train, Ytrain):
      print Xtrain.shape
      print labels_train.shape
      model.fit(Xtrain, labels_train)
      predictions = model.predict(Ytrain)
      return predictions
      
   def trainfit(self, model, Xtrain, labels_train):
      model.fit(Xtrain, labels_train)
      train_predictions = model.predict(Xtrain)
      return train_predictions

   def print_crossval(self, model, Xtrain, labels_train):
      trainfit = cross_val_score(model, Xtrain, labels_train)
      print "Score = %s"%(trainfit)

   def score(self, X, Y):
      train_predictions = self.trainfit(model, X, Y) 
      d1 = (Y - train_predicitons.reshape(len(X),1)).ravel()
      d2 = (Y - Y.mean()).ravel()
      return 1 - d1.dot(d1) / d2.dot(d2)

   def plot_data(self, modelname, Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions):
      plt.title(modelname + " : Train performance")
      plt.scatter(Xtrain, labels_train, c = 'r')
      plt.scatter(Xtrain, train_predictions, c = 'b')
      plt.show()
      plt.title(modelname + " : Test performance")
      plt.scatter(Ytrain, labels_test, c = 'r')
      plt.scatter(Ytrain, predictions, c = 'b')
      plt.show()
 
class BaggedRegressor:
    def __init__(self, learn_model, B, bag_size):
        self.B = B
        self.bag_size = bag_size
        self.learnmodel = learn_model
    def fit(self, X, Y):
        N = len(X)
        self.models = []
        for b in xrange(self.B):
            idx = np.random.choice(N, size=self.bag_size, replace=True)
            Xb = X[idx].reshape(self.bag_size,1)
            Yb = Y[idx].reshape(self.bag_size,1)
            if self.learnmodel == 'DecisionTreeRegressor':
                learn_model = DecisionTreeRegressor()
            if self.learnmodel == 'RandomForestRegressor':
                learn_model = RandomForestRegressor()
            #learn_model = self.learnmodel 
            learn_model.fit(Xb, Yb)
            self.models.append(learn_model)
    def predict(self, X):
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X)
        return predictions / self.B
    def score(self, X, Y):
        d1 = (Y - self.predict(X).reshape(len(X),1)).ravel()
        d2 = (Y - Y.mean()).ravel()
        return 1 - d1.dot(d1) / d2.dot(d2)

if __name__ == '__main__':
   obj = datamodel()
   N=100
   split_size=80
   Xtrain, Ytrain, labels_train, labels_test = obj.generate_sample_data(split_size, N)

   #Model : Decision Tree
   model = DecisionTreeRegressor(max_depth=10)
   predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   train_predictions = obj.trainfit(model, Xtrain, labels_train)
   #obj.print_crossval(model, Xtrain, labels_train) 
   print "Train Score for 1 Decision Tree : %s"%(model.score(Xtrain, labels_train))
   print "Test Score for 1 Decision Tree : %s"%(model.score(Ytrain, labels_test))
   obj.plot_data("Decision Tree", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)
   
   #Model : Random Forest Regressor 
   model = RandomForestRegressor()
   predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   train_predictions = obj.trainfit(model, Xtrain, labels_train)
   #obj.print_crossval(model, Xtrain, labels_train) 
   print "Train Score for random forest : %s"%(model.score(Xtrain, labels_train))
   print "Test Score for random forest : %s"%(model.score(Ytrain, labels_test))
   obj.plot_data("Random Forest", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)

   #Model : Bagged Decision Tree Regressor
   model = BaggedRegressor('DecisionTreeRegressor',200,80)
   model.fit(Xtrain.ravel(), labels_train.ravel())
   predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   train_predictions = obj.trainfit(model, Xtrain, labels_train)
   print "Train Score for bagged trees : %s"%(model.score(Xtrain, labels_train))
   print "Test Score for bagged trees : %s"%(model.score(Ytrain, labels_test))
   obj.plot_data("Bagged DT : ", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)

   #Model : Data and Feature Bagged (RF) Regressor
   model = BaggedRegressor('RandomForestRegressor',200,80)
   model.fit(Xtrain.ravel(), labels_train.ravel())
   predictions = obj.modelfit(model, Xtrain, labels_train, Ytrain)
   train_predictions = obj.trainfit(model, Xtrain, labels_train)
   print "Train Score for bagged RF trees : %s"%(model.score(Xtrain, labels_train))
   print "Test Score for bagged RF trees : %s"%(model.score(Ytrain, labels_test))
   obj.plot_data("Bagged RF DT : ", Xtrain, Ytrain, labels_train, labels_test, predictions, train_predictions)


