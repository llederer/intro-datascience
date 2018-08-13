"""
Recommendation Systems
    Use machine learning to go through all options and
    learn what we like
    1. Collaborative Systems:
        predicts what we like based on what similar users like
    2. Content-Based System:
        predicts what you like based on what you have already liked

Dependencies:
    Numpy
    SciPy
    Lightfm
        fetch_movielens: data includes 100k movie ratings (CSV file) from 1k users
            each user has rated at least 20 movies
        
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=4.0) #min_rating is an optional parameter

#We are only collecting movies with a rating of 4.0 or higher, we will be storing
#this data in a dictionary (stores data using any variable, in this case, string)

#printtraining and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model 
#warp = Weighted Approximate-Rank Pairwise
model = LightFM(loss = 'warp')

#train model
model.fit(data['train'], epochs=30, num_threads = 2)
"""
epochs = runs
threads = parallel computations
"""

#creating a recommendation function

def sample_recommendation(model, data, user_ids):
    
    #number of users and movies in training data
    n_users, n_items = data['train'].shape
    
    #generate recommendations for each user we input (for loop)
    for user_id in user_ids:
        
        #movies they already like
        #compressed sparse row format
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        
        #print out the results
        print("User %s" % user_id)
        print("      Known positives:" )
        
        for x in known_positives[:3]:
            print("         %s" % x)
        
        print("      Recommended:")
        
        for x in top_items[:3]:
            print("         %s" % x)
            

sample_recommendation(model, data, [3, 25, 450])