import pickle
# load : get the data from file
path = "models/RandomForest.pkl"
fileobj = pickle.load(open(path , 'rb'))
# loads : get the data from var.
sd = pickle.load(fileobj)
print(type(sd))