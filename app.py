import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import tree
from sklearn.linear_model import SGDRegressor

games = pandas.read_csv("./board_games.csv")
games = games[games["users_rated"] > 0]
games = games.dropna(axis=0)

kmeans_model = KMeans(n_clusters = 5, random_state=1)
# Get only the numeric columns from games.
good_columns = games._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(good_columns)
# Get the cluster assignments.
labels = kmeans_model.labels_

pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0],y=plot_columns[:,1], c=labels)
plt.show()

columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name"]]
target = "average_rating"

train = games.sample(frac=0.8, random_state=1)
test = games.loc[~games.index.isin(train.index)]

model = LinearRegression()
model.fit(train[columns], train[target])

predictions = model.predict(test[columns])
print (mean_squared_error(predictions, test[target]))

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
print (mean_squared_error(predictions, test[target]))

model = SVR()
model.fit(train[columns], train[target])
predictions = model.predict(test[columns])
print (mean_squared_error(predictions, test[target]))

model = tree.DecisionTreeRegressor()
model.fit(train[columns], train[target])
predictions = model.predict(test[columns])
print (mean_squared_error(predictions, test[target]))

