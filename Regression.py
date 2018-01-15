#################### Implementing Different Regression Models to Predict Page Load Time ####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from math import sqrt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from xgboost.sklearn import XGBRegressor
from scipy.stats import skew

with open("./latest_stats/filtered_latest_all_no_outlier_stat.csv") as f:
	filtered_all_no_outlier_stat_content = f.readlines()

reg_data = open("Reg-Features.csv", "w")
for line in filtered_all_no_outlier_stat_content:
			
	#if float(line.split(' ')[1]) < 100:

	######### Page Load Time
	reg_data.write(line.split(' ')[1])
	reg_data.write(",")

	######### Page Total Size
	if float(line.split(' ')[2]) < 80000000:
		reg_data.write(line.split(' ')[2])
		reg_data.write(",")
	else:
		reg_data.write("0")
		reg_data.write(",")

	######### Objects Requested
	reg_data.write(line.split(' ')[3])
	reg_data.write(",")


	######### All Objects Returned
	reg_data.write(str(float(line.split(' ')[4]) + float(line.split(' ')[5]) + float(line.split(' ')[6])))
	reg_data.write(",")

	######### Objects Returned Origin
	reg_data.write(str(float(line.split(' ')[4])))
	reg_data.write(",")

	######### Objects Returned NonOrigin
	reg_data.write(str(float(line.split(' ')[5])))
	reg_data.write(",")

	######### All Servers
	reg_data.write(str(float(line.split(' ')[7]) + float(line.split(' ')[8]) + float(line.split(' ')[9])))
	reg_data.write(",")

	######### Origin Servers
	reg_data.write(str(line.split(' ')[7]))
	reg_data.write(",")

	######### Non-Origin Servers
	reg_data.write(str(line.split(' ')[8]))
	reg_data.write(",")

	######### JS Number
	reg_data.write(line.split(' ')[10])
	reg_data.write(",")

	######### JS Size
	reg_data.write(line.split(' ')[11])
	reg_data.write(",")

	######### HTML Number
	reg_data.write(line.split(' ')[16])
	reg_data.write(",")

	######### CSS Number
	reg_data.write(line.split(' ')[22])
	reg_data.write(",")

	######### Image Number
	reg_data.write(line.split(' ')[28])
	reg_data.write(",")

	######### Image Size
	reg_data.write(line.split(' ')[29])
	#reg_data.write(" ")

	reg_data.write("\n")
				

reg_data.close()

#####*************** Reading the Features and Load Times **********************#####

column_names = ["LoadTime", "PageSize", "ReqObjs", "RetObjs", "RetObjsOrig", "RetObjsNonOrig", "Servers", "OrigSer", "NonOrigSer", "JSNum", "JSSize", "HTMLNum", "CSSNum", "ImageNum", "ImageSize"]
#feature_cols = ["PageSize", "ReqObjs", "RetObjs", "Servers", "OrigSer", "NonOrigSer", "JSNum", "JSSize", "HTMLNum", "CSSNum", "ImageNum", "ImageSize"]
data = pd.read_csv("Reg-Features.csv", names=column_names)
#print data.head()
#features = data.values[:, 1:4]
#load_time = data.values[:, 0]
#print features.shape


#####*************** Exploring the Page Load Time **********************#####
sns.distplot(data['LoadTime'])
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
load_times = pd.DataFrame({"loadTime":data["LoadTime"], "log(loadTime + 1)":np.log1p(data["LoadTime"])})
load_times.hist()
print "Load Time Description: ", data['LoadTime'].describe()
print data.loc[(data.LoadTime > 100), 'LoadTime'].count()
#print type(load_time)
#print load_time.shape

####### Standardizing Load Time #######
loadTime_scaled = preprocessing.StandardScaler().fit_transform(data['LoadTime'][:,np.newaxis]);
low_range = loadTime_scaled[loadTime_scaled[:,0].argsort()][:10]
high_range = loadTime_scaled[loadTime_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)


#####*************** Initial Feature Skewness *****************#####
skewness = data.apply(lambda x: skew(x))
print skewness

#####*************** Feature Engineering **********************#####

# poly = PolynomialFeatures(degree=2)
# poly_features = poly.fit_transform(features)
# normalized_features = preprocessing.normalize(features, norm='l1')

data['PercNonOrigSer'] = data['NonOrigSer'] / data['Servers']
data['PercNonOrigObj'] = data['RetObjsNonOrig'] / data['RetObjs']
data['PercOrigObj'] = data['RetObjsOrig'] / data['RetObjs']
data.loc[~np.isfinite(data['PercNonOrigSer']), 'PercNonOrigSer'] = 0
data.loc[~np.isfinite(data['PercNonOrigObj']), 'PercNonOrigObj'] = 0
data.loc[~np.isfinite(data['PercOrigObj']), 'PercOrigObj'] = 0

# data['LogPageSize'] = np.log1p(data['PageSize'] + 0.000001)
# data['LogJSSize'] = np.log1p(data['JSSize'] + 0.000001)
# data['LogJSNum'] = np.log1p(data['JSNum'] + 0.000001)
# data['LogReqObjs'] = np.log1p(data['ReqObjs'] + 0.000001)
# data['LogRetObjs'] = np.log1p(data['RetObjs'] + 0.000001)
# data['LogServers'] = np.log1p(data['Servers'] + 0.000001)
# data['LogImageNum'] = np.log1p(data['ImageNum'] + 0.000001)
# data['LogImageSize'] = np.log1p(data['ImageSize'] + 0.000001)
# data['LogOrigSer'] = np.log1p(data['OrigSer'] + 0.000001)
# data['LogNonOrigSer'] = np.log1p(data['NonOrigSer'] + 0.000001)
# data['LogHTMLNum'] = np.log1p(data['HTMLNum'] + 0.000001)
# data['LogCSSNum'] = np.log1p(data['CSSNum'] + 0.000001)
# data['LogPercNonOrigSer'] = np.log1p(data['PercNonOrigSer'] + 0.000001)
# data['LogPercNonOrigObj'] = np.log1p(data['PercNonOrigObj'] + 0.000001)
# data['LogPercOrigObj'] = np.log1p(data['PercOrigObj'] + 0.000001)

# data['PageSize'] = np.log1p(data['PageSize'] + 0.000001)
# data['JSSize'] = np.log1p(data['JSSize'] + 0.000001)
# data['JSNum'] = np.log1p(data['JSNum'] + 0.000001)
# data['ReqObjs'] = np.log1p(data['ReqObjs'] + 0.000001)
# data['RetObjs'] = np.log1p(data['RetObjs'] + 0.000001)
# data['RetObjsNonOrig'] = np.log1p(data['RetObjsNonOrig'] + 0.000001)
# data['RetObjsOrig'] = np.log1p(data['RetObjsOrig'] + 0.000001)
# data['Servers'] = np.log1p(data['Servers'] + 0.000001)
# data['ImageNum'] = np.log1p(data['ImageNum'] + 0.000001)
# data['ImageSize'] = np.log1p(data['ImageSize'] + 0.000001)
# data['OrigSer'] = np.log1p(data['OrigSer'] + 0.000001)
# data['NonOrigSer'] = np.log1p(data['NonOrigSer'] + 0.000001)
# data['HTMLNum'] = np.log1p(data['HTMLNum'] + 0.000001)
# data['CSSNum'] = np.log1p(data['CSSNum'] + 0.000001)

# data["PageSize-2"] = data["PageSize"] ** 2
# data["ReqObjs-2"] = data["ReqObjs"] ** 2
# data["RetObjs-2"] = data["RetObjs"] ** 2
# data["Servers-2"] = data["Servers"] ** 2
# data["OrigSer-2"] = data["OrigSer"] ** 2
# data["NonOrigSer-2"] = data["NonOrigSer"] ** 2
# data["JSNum-2"] = data["JSNum"] ** 2
# data["JSSize-2"] = data["JSSize"] ** 2
# data["ImageNum-2"] = data["ImageNum"] ** 2
# data["ImageSize-2"] = data["ImageSize"] ** 2
# data["HTMLNum-2"] = data["HTMLNum"] ** 2
# data["CSSNum-2"] = data["CSSNum"] ** 2
# data['PercNonOrigSer-2'] = data['PercNonOrigSer'] ** 2
# data['PercNonOrigObj-2'] = data['PercNonOrigObj'] ** 2
# data['PercOrigObj-2'] = data['PercOrigObj'] ** 2

#data["LoadTime"] = np.log1p(data["LoadTime"])

#features = data[feature_cols]
data_cols = data.columns
scale_features = data_cols.drop(["LoadTime", "PercNonOrigSer", "PercOrigObj", "PercNonOrigObj", "RetObjsOrig", "OrigSer"])
feature_cols = data_cols.drop(["LoadTime", "PercOrigObj","RetObjsOrig", "OrigSer"]) 
								#"CSSNum", "PercNonOrigObj", "PercNonOrigSer", "PageSize"])
features = data[feature_cols]
load_time = data['LoadTime']


#####*************** Feature Scaling and Handling Skewness **********************#####

#scaler = preprocessing.Normalizer()
scaler = preprocessing.RobustScaler()
#scaler = MinMaxScaler(feature_range=(0, 1))
features[scale_features] = scaler.fit_transform(features[scale_features])
features.hist()
plt.show()

#log transform skewed numeric features:
#numeric_feats = data.dtypes[data.dtypes != "object"].index
#skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#data[skewed_feats].apply(lambda x: np.log1p(x + 0.00001))
#print "Skewed Features: ", data[skewed_feats]
print features['RetObjs'].describe()
print "New Skewness: ", features.skew()


#####*************** Feature Correlations **********************#####

###### Correlation Matrix ######
corrmat = data.corr()

###### Load Time correlation matrix ######
k = 14 # number of variables for heatmap
cols = corrmat.nlargest(k, 'LoadTime')['LoadTime'].index
cm = np.corrcoef(data[cols].values.T)
print "Correlations: ", corrmat.nlargest(k, 'LoadTime')['LoadTime']
sns.set(font_scale=0.8)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#####*************** Scatter Plots **********************#####
fig, axes = plt.subplots(nrows=4, ncols=3)
data.plot(ax=axes[0,0], kind='scatter',x='PageSize', y='LoadTime')
data.plot(ax=axes[0,1], kind='scatter',x='ReqObjs', y='LoadTime')
data.plot(ax=axes[0,2], kind='scatter',x='RetObjs', y='LoadTime')
data.plot(ax=axes[1,2], kind='scatter',x='Servers', y='LoadTime')
data.plot(ax=axes[1,0], kind='scatter',x='JSNum', y='LoadTime')
data.plot(ax=axes[1,1], kind='scatter',x='ImageNum', y='LoadTime')
data.plot(ax=axes[2,0], kind='scatter',x='JSSize', y='LoadTime')
data.plot(ax=axes[2,1], kind='scatter',x='ImageSize', y='LoadTime')
data.plot(ax=axes[3,0], kind='scatter',x='PercNonOrigObj', y='LoadTime')
data.plot(ax=axes[3,1], kind='scatter',x='PercOrigObj', y='LoadTime')
data.plot(ax=axes[3,2], kind='scatter',x='PercNonOrigSer', y='LoadTime')
plt.show()



#####*************** Regression Models **********************#####

regr = LinearRegression()
#regr = linear_model.Lasso(alpha=0.1)
#regr = linear_model.Ridge(alpha=0.01)
#regr = SVR()

######################### Ridge Regression #########################
def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model, features, load_time, scoring="neg_mean_squared_error", cv = 10))
	return(rmse)

def r2_score_cv(model):
	r2_score = cross_val_score(model, features, load_time, scoring="r2", cv = 10)
	return(r2_score)

model_ridge = Ridge()
ridge_predicted = cross_val_predict(Ridge(alpha = 10), features, load_time, cv=10)
alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge_r2_score = [r2_score_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()


######################### Lasso Regression #########################
#model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(features, load_time)
#model_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
min_lasso_error = 1000000
for alpha in alphas:
	lasso_error = rmse_cv(Lasso(alpha = alpha)).mean()
	if lasso_error < min_lasso_error:
		min_lasso_error = lasso_error
		min_alpha = alpha

best_lasso_model = Lasso(min_alpha).fit(features, load_time)
lasso_r2_scores = cross_val_score(best_lasso_model, features, load_time, scoring='r2', cv=10)
coef = pd.Series(best_lasso_model.coef_, index = feature_cols)
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


######################### ElasticNet Regression #########################
ElasticNet(random_state=0)
cv_elasticNet = [rmse_cv(ElasticNet(alpha = alpha, random_state=0)).mean() for alpha in alphas]
cv_elasticNet_r2_score = [r2_score_cv(ElasticNet(alpha = alpha, random_state=0)).mean() for alpha in alphas]
cv_elasticNet = pd.Series(cv_elasticNet)

######################### XGBoost Regression #########################
xgb1 = XGBRegressor(colsample_bytree=0.2,
                 learning_rate=0.05,
                 max_depth=3,
                 n_estimators=1200
                )

xgboost_cv_rmse = np.sqrt(-cross_val_score(xgb1, features, load_time, cv = 10, n_jobs=-1, scoring='neg_mean_squared_error'))


###### Splitting Data into Train and Test data ######
X_train, X_test, y_train, y_test = train_test_split(features, load_time, test_size=0.2)


######################### Linear Regression #########################

predicted = cross_val_predict(regr, features, load_time, cv=10)

mean_square_scores = cross_val_score(regr, features, load_time, scoring='neg_mean_squared_error', cv=10)
rms = sqrt(abs(mean_square_scores.mean()))
linear_reg_r2_scores = cross_val_score(regr, features, load_time, scoring='r2', cv=10)
regr.fit(X_train, y_train)

######### Linear Regression Feature Importances #########
print "Linear Regression Features Coefficients: "
for coefficient in list(zip(feature_cols, regr.coef_)):
	print coefficient

######### Prediction Plot #########
fig, ax = plt.subplots()
ax.scatter(load_time, ridge_predicted, edgecolors=(0, 0, 0))
ax.plot([load_time.min(), load_time.max()], [load_time.min(), load_time.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


######### Printing Results #########

load_time_range = data['LoadTime'].max() - data['LoadTime'].min()
print "-----------------------------------------------------------------"
print "Linear Regression Load Time Prediction: ", predicted.mean()
print "Linear Regression Root Mean Square Error: ", rms
print "Linear Regression Normalized Root Mean Square Error: ", float(rms) / load_time_range
print "Linear Regression R2 Score: ", linear_reg_r2_scores.mean()

print "***************************************************"
print "XGBoost Root Mean Square Error: ", xgboost_cv_rmse.mean()
print "***************************************************"
print "Ridge Root Mean Square Error: ", cv_ridge.min()
print "Ridge Normalized Root Mean Square Error: ", cv_ridge.min() / load_time_range
print "Ridge R2 Score: ", max(cv_ridge_r2_score)
print "***************************************************"
print "Lasso Root Mean Square Error: ", min_lasso_error
print "Lasso Normalized Root Mean Square Error: ", min_lasso_error / load_time_range
print "Lasso R2 Score: ", lasso_r2_scores.mean()
print "***************************************************"
print "ElasticNet Root Mean Square Error: ", cv_elasticNet.min()
print "ElasticNet R2 Score: ", max(cv_elasticNet_r2_score)








