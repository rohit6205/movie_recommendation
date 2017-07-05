import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import lightFM 

#fetch data
data = fetch_movielens(min_rating = 4.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp')

model1 = LightFM(loss='logistic')

model2 = LightFM(loss='warp-kos')

model3 = LightFM(loss='bpr')
#train model
model.fit(data[train],epochs=30,num_threads=2)

model1.fit(data[train],epochs=30,num_threads=2)

model2.fit(data[train],epochs=30,num_threads=2)

model3.fit(data[train],epochs=30,num_threads=2)

def sample_recommendation(model, data, user_ids):

	#number of users and movies in training data
	n_users, n_items = data['train'].shape

	#generate recommendation for each user we input
	for user_id in user_ids	:

		#movies they already like
		known_positives = data['item_labels'] [data['train'].tocsr()[user_id].indices]

		#movies our model predict they will like
		score  = model.predict(user_id, np.arange(n_items))
		#rank them in order of most liked to least
		top_items = data['item_labels'] [np.argsort(-score)]

		#print out the result
		print("user %s" % user_id)
		print("        known positives:")

		for x in known_positives[:3]:
			print("          %s" % x)

		print("        recommended:")

		for x in top_items[:3]:
			print("        %s" % x)


sample_recommendation(model, data, [3,25,450])