from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

X, y = datasets.load_breast_cancer(return_X_y=True) 

print("There are",X.shape[0], "instances described by", X.shape[1], "features.") 


X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.4, stratify=y, random_state = 42)  #(5 points) 


clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=6)  
clf.fit(X_train, y_train)   


predC = clf.predict(X_test)
 
print('The accuracy of the classifier is', accuracy_score(y_test,predC))  

_ = tree.plot_tree(clf,filled=True, fontsize=12)  


trainAccuracy = []  
testAccuracy = []

depthOptions = range(1,16) 
for depth in depthOptions:      
    cltree = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=6,max_depth=depth) 
    
    cltree.fit(X_train,y_train)  
    
    y_predTrain = cltree.predict(X_train)  
    
    y_predTest = cltree.predict(X_test) 
   
    trainAccuracy.append(accuracy_score(y_train,y_predTrain)) 
    
    testAccuracy.append(accuracy_score(y_test,y_predTest))  


plt.plot(depthOptions,trainAccuracy,'ro-', label = "Training Accuracy")
plt.plot(depthOptions,testAccuracy,'bx-', label = "Testing Accuracy") 
plt.legend(['Training Accuracy','Test Accuracy']) 
plt.ylabel('Classifier Accuracy')  




parameters = {'max_depth': range(1,16)} 

clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=6, random_state=42),parameters)#(6 points)
clf.fit(X_train, y_train) 
tree_model = clf.best_estimator_ 
print("The maximum depth of the tree should be", clf.best_params_['max_depth']) 


_ = tree.plot_tree(tree_model,filled=True, fontsize=12) 




