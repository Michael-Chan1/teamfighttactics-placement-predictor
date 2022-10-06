import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

# get dummy data 
df = pd.read_csv("game_outcomes.csv")

# train test split 
from sklearn.model_selection import train_test_split
df = pd.read_csv("game_outcomes.csv")
num_games = df.iloc[df.index[-1]]['game']
df_train = df.loc[df['game'] % 5 != 0]
df_test = df.loc[df['game'] % 5 == 0]
y = df_train.placement.values
X = pd.get_dummies(df_train.drop(['placement', 'game'], axis =1))
y_test = df_test.placement.values
y_test_games = df_test.game.values
X_test = pd.get_dummies(df_test.drop(['placement', 'game'], axis =1))

from sklearn.utils import shuffle
X_train, y_train = shuffle(X, y, random_state=0)

# random forest 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

np.mean(cross_val_score(rfr,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3))

# tune models GridsearchCV 
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

rf = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
rf.fit(X_train,y_train)

num_incorrect = 0
games = 0
for i in range(5, num_games, 5):
    current_placement = 0
    indicies = np.where(y_test_games == i)
    if (y[indicies] == 1).sum() == 1:
        p1 = rf.predict(X_test[np.where(y[indicies] == 1)[0][0]:np.where(y[indicies] == 1)[0][-1] + 1]).sum()/(y[indicies] == 1).sum()
    elif (y[indicies] == 1).sum() == 0:
        p1 = 99999
    else:
        p1 = rf.predict(X_test[np.where(y[indicies] == 1)[0][0]:np.where(y[indicies] == 1)[0][-1]]).sum()/(y[indicies] == 1).sum()
    
    if (y[indicies] == 2).sum() == 1:
        p2 = rf.predict(X_test[np.where(y[indicies] == 2)[0][0]:np.where(y[indicies] == 2)[0][-1] + 1]).sum()/(y[indicies] == 2).sum()
    elif (y[indicies] == 2).sum() == 0:
        p2 = 99999
    else:
        p2 = rf.predict(X_test[np.where(y[indicies] == 2)[0][0]:np.where(y[indicies] == 2)[0][-1]]).sum()/(y[indicies] == 2).sum()
        
    if (y[indicies] == 3).sum() == 1:
        p3 = rf.predict(X_test[np.where(y[indicies] == 3)[0][0]:np.where(y[indicies] == 3)[0][-1] + 1]).sum()/(y[indicies] == 3).sum()
    elif (y[indicies] == 3).sum() == 0:
        p3 = 99999
    else:
        p3 = rf.predict(X_test[np.where(y[indicies] == 3)[0][0]:np.where(y[indicies] == 3)[0][-1]]).sum()/(y[indicies] == 3).sum()
        
    if (y[indicies] == 4).sum() == 1:
        p4 = rf.predict(X_test[np.where(y[indicies] == 4)[0][0]:np.where(y[indicies] == 4)[0][-1] + 1]).sum()/(y[indicies] == 4).sum()
    elif (y[indicies] == 4).sum() == 0:
        p4 = 99999
    else:
        p4 = rf.predict(X_test[np.where(y[indicies] == 4)[0][0]:np.where(y[indicies] == 4)[0][-1]]).sum()/(y[indicies] == 4).sum()
                               
    if (y[indicies] == 5).sum() == 1:
        p5 = rf.predict(X_test[np.where(y[indicies] == 5)[0][0]:np.where(y[indicies] == 5)[0][-1] + 1]).sum()/(y[indicies] == 5).sum()
    elif (y[indicies] == 5).sum() == 0:
        p5 = 99999
    else:
        p5 = rf.predict(X_test[np.where(y[indicies] == 5)[0][0]:np.where(y[indicies] == 5)[0][-1]]).sum()/(y[indicies] == 5).sum()
                               
    if (y[indicies] == 6).sum() == 1:
        p6 = rf.predict(X_test[np.where(y[indicies] == 6)[0][0]:np.where(y[indicies] == 6)[0][-1] + 1]).sum()/(y[indicies] == 6).sum()
    elif (y[indicies] == 6).sum() == 0:
        p6 = 99999
    else:                        
        p6 = rf.predict(X_test[np.where(y[indicies] == 6)[0][0]:np.where(y[indicies] == 6)[0][-1]]).sum()/(y[indicies] == 6).sum()
                               
    if (y[indicies] == 7).sum() == 1:
        p7 = rf.predict(X_test[np.where(y[indicies] == 7)[0][0]:np.where(y[indicies] == 7)[0][-1] + 1]).sum()/(y[indicies] == 7).sum()
    elif (y[indicies] == 7).sum() == 0:
        p7 = 99999
    else:
        p7 = rf.predict(X_test[np.where(y[indicies] == 7)[0][0]:np.where(y[indicies] == 7)[0][-1]]).sum()/(y[indicies] == 7).sum()
                               
    if (y[indicies] == 8).sum() == 1:
        p8 = rf.predict(X_test[np.where(y[indicies] == 8)[0][0]:np.where(y[indicies] == 8)[0][-1] + 1]).sum()/(y[indicies] == 8).sum()
    elif (y[indicies] == 8).sum() == 0:
        p8 = 99999
    else:
        p8 = rf.predict(X_test[np.where(y[indicies] == 8)[0][0]:np.where(y[indicies] == 8)[0][-1]]).sum()/(y[indicies] == 8).sum()
    predicted_placements = [0, p1, p2, p3, p4, p5, p6, p7, p8]
    predicted_placements.sort()
    num_incorrect += (abs(predicted_placements.index(p1) - 1) + abs(predicted_placements.index(p2) - 2) + abs(predicted_placements.index(p3) - 3) + abs(predicted_placements.index(p4) - 4) + abs(predicted_placements.index(p5) - 5) + abs(predicted_placements.index(p6) - 6) + abs(predicted_placements.index(p7) - 7) + abs(predicted_placements.index(p8) - 8))/2
    games += 1
print("Accuracy = " + str(1 - num_incorrect/(36 * games)))
        
        
        
        
        
        
        
        
        
        