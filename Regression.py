
#################### Implementing different regression models to predict the page load times of our target websites ####################
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, Normalizer, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
from scipy.stats import spearmanr, pearsonr
from math import sqrt
from xgboost.sklearn import XGBRegressor
from scipy.stats import skew


#####*************** Reading the complexity metrics and page load times from the csv files **********************#####

######## Defining column names ########
timing_column_names = ["PageURL", "ReqObjs", "MeanObjectsSize", "LoadTime", "MaxDelay", "MaxTimingValue", "MaxTimingfactor", "MaxBlockedTime",
                "MaxDNSTime", "MaxConncetTime", "MaxSendTime", "MaxWaitTime", "MaxReceiveTime", "ServerMaxTime",
                "AllMaxLess2", "AllMax23", "AllMax34", "AllMax45", "AllMax56", "AllMax67", "AllMax78", "AllMax89", "AllMax910", "MaxGap", "MaxGapServer"]

stat_column_names = ["PageURL", "LoadTime", "PageSize", "ReqObjs", "NumObjsOrigin", "NumObjsNonOrigin",
                     "NumObjsUnknown", "NumOriginServers", "NonOrigSer", "NumUnknownServers", "JSNum",
                     "JSSize", "num_origin_javascript_matches", "size_origin_javascript",
                     "num_non_origin_javascript_matches", "size_non_origin_javascript", "HTMLNum",
                     "size_html", "num_origin_html_matches", "size_origin_html", "num_non_origin_html_matches",
                     "size_non_origin_html", "CSSNum", "size_css", "num_origin_css_matches",
                     "size_origin_css", "num_non_origin_css_matches", "size_non_origin_css", "ImageNum",
                     "size_image", "num_origin_image_matches", "size_origin_image", "num_non_origin_image_matches",
                     "size_non_origin_image", "num_xml_matches", "size_xml", "num_origin_xml_matches",
                     "size_origin_xml", "num_non_origin_xml_matches", "size_non_origin_xml", "num_plain_text_matches",
                     "size_plain_text", "num_origin_plain_text_matches", "size_origin_plain_text",
                     "num_non_origin_plain_text_matches", "size_non_origin_plain_text", "num_json_matches",
                     "size_json", "num_origin_json_matches", "size_origin_json", "num_non_origin_json_matches",
                     "size_non_origin_json", "num_flash_matches", "size_flash", "num_origin_flash_matches",
                     "size_origin_flash", "num_non_origin_flash_matches", "size_non_origin_flash",
                     "num_font_matches", "size_font", "num_origin_font_matches", "size_origin_font",
                     "num_non_origin_font_matches", "size_non_origin_font", "num_audio_matches",
                     "size_audio", "num_origin_audio_matches", "size_origin_audio",
                     "num_non_origin_audio_matches", "size_non_origin_audio", "num_video_matches",
                     "size_video", "num_origin_video_matches", "size_origin_video", 
                     "num_non_origin_video_matches", "size_non_origin_video", "num_other",
                     "size_other", "num_origin_other", "size_origin_other", "num_non_origin_other",
                     "size_non_origin_other", "num_no_type"]


selected_columns_VPs = ['PageURL', 'PageSize', 'LoadTime', 'VP', 'ReqObjs', 'RetObjs', 'Servers', 'NonOrigSer', 'JSNum',
                     'ImageNum', 'HTMLNum', 'CSSNum', 'JSSize', 'NumOriginServers', 'NumUnknownServers',
                     'NumObjsOrigin', 'NumObjsNonOrigin', 'NumObjsUnknown']


######## Reading csv files ########
eugene_timing_data = pd.read_csv("eugene_time_results.csv", sep=' ', names=timing_column_names)
china_timing_data = pd.read_csv("china_time_results.csv", sep=' ', names=timing_column_names)
brazil_timing_data = pd.read_csv("brazil_time_results.csv", sep=' ', names=timing_column_names)
ny_timing_data = pd.read_csv("ny_time_results.csv", sep=' ', names=timing_column_names)
spain_timing_data = pd.read_csv("spain_time_results.csv", sep=' ', names=timing_column_names)

eugene_stat_data = pd.read_csv("eugene_stat_data.csv", sep=' ', names=stat_column_names)
brazil_stat_data = pd.read_csv("brazil_stat_data.csv", sep=' ', names=stat_column_names)
china_stat_data = pd.read_csv("china_stat_data.csv", sep=' ', names=stat_column_names)
spain_stat_data = pd.read_csv("spain_stat_data.csv", sep=' ', names=stat_column_names)
ny_stat_data = pd.read_csv("ny_stat_data.csv", sep=' ', names=stat_column_names)

######## Selecting the desired columns from the stat data ########
eugene_stat_data = eugene_stat_data.loc[:, selected_columns_VPs]
china_stat_data = china_stat_data.loc[:, selected_columns_VPs]
brazil_stat_data = brazil_stat_data.loc[:, selected_columns_VPs]
ny_stat_data = ny_stat_data.loc[:, selected_columns_VPs]
spain_stat_data = spain_stat_data.loc[:, selected_columns_VPs]

######## Selecting the desired columns from the timing data ########
eugene_timing_data = eugene_timing_data.loc[:, ['PageURL', 'MaxDelay']]
china_timing_data = china_timing_data.loc[:, ['PageURL', 'MaxDelay']]
brazil_timing_data = brazil_timing_data.loc[:, ['PageURL', 'MaxDelay']]
ny_timing_data = ny_timing_data.loc[:, ['PageURL', 'MaxDelay']]
spain_timing_data = spain_timing_data.loc[:, ['PageURL', 'MaxDelay']]

######## Merging timing values with stat values to add the MaxDelay column ########
merged_eugene = eugene_stat_data.merge(eugene_timing_data, on=['PageURL'])
merged_china = china_stat_data.merge(china_timing_data, on=['PageURL'])
merged_brazil = brazil_stat_data.merge(brazil_timing_data, on=['PageURL'])
merged_ny = ny_stat_data.merge(ny_timing_data, on=['PageURL'])
merged_spain = spain_stat_data.merge(spain_timing_data, on=['PageURL'])

######## Removing duplicates ########
merged_eugene = merged_eugene.drop_duplicates(subset=['PageURL', 'LoadTime'])
merged_china = merged_china.drop_duplicates(subset=['PageURL', 'LoadTime'])
merged_brazil = merged_brazil.drop_duplicates(subset=['PageURL', 'LoadTime'])
merged_ny = merged_ny.drop_duplicates(subset=['PageURL', 'LoadTime'])
merged_spain = merged_spain.drop_duplicates(subset=['PageURL', 'LoadTime'])

######## Adding the vantage point (VP) column ########
merged_eugene['VP'] = "Eugene"
merged_china['VP'] = "China"
merged_brazil['VP'] = "Brazil"
merged_ny['VP'] = "NY"
merged_spain['VP'] = "Spain"

######## Concatenating all the data entries from different VPs ########
all_stat_data = [merged_china, merged_eugene, merged_brazil, merged_ny, merged_spain]
all_stat_data = pd.concat(all_stat_data)

######## Combining some of the columns to add new features ########
all_stat_data['Servers'] = all_stat_data.NonOrigSer + all_stat_data.NumOriginServers + all_stat_data.NumUnknownServers
all_stat_data['RetObjs'] = all_stat_data.NumObjsOrigin + all_stat_data.NumObjsNonOrigin + all_stat_data.NumObjsUnknown
all_stat_data['AllMax1'] = all_stat_data.LoadTime / (all_stat_data.MaxDelay / 1000)

######## Removing outliers ########
all_stat_data = all_stat_data.loc[((all_stat_data.AllMax1 > 2) | (all_stat_data.LoadTime < 10))]

######## Selecting columns needed for the regression ########
selected_columns_data = ['LoadTime', 'PageSize', 'VP', 'ReqObjs', 'RetObjs', 'Servers', 'NonOrigSer', 'JSNum',
                     'ImageNum', 'HTMLNum', 'CSSNum', 'JSSize', 'MaxDelay', 'AllMax1', 'NumObjsNonOrigin']
data = all_stat_data.loc[:, selected_columns_data]

#####*************** Looking into the page load times **********************#####
### Distribution of page load times
sns.distplot(data['LoadTime'])
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.show()

loadTime_scaled = StandardScaler().fit_transform(data['LoadTime'][:,np.newaxis]);
low_range = loadTime_scaled[loadTime_scaled[:, 0].argsort()][:10]
high_range = loadTime_scaled[loadTime_scaled[:, 0].argsort()][-10:]
print "outer range (low) of the distribution:\n", low_range
print "outer range (high) of the distribution:\n", high_range

### log transformation of the page load times
load_times = pd.DataFrame({"loadTime":data["LoadTime"], "log(loadTime + 1)":np.log1p(data["LoadTime"])})
load_times.hist()
plt.show()

print "Load Time Description:\n", data['LoadTime'].describe()
#print data.loc[(data.LoadTime > 100), 'LoadTime'].count()

#####*************** Feature Engineering **********************#####
# poly = PolynomialFeatures(degree=2)
# poly_features = poly.fit_transform(features)

# data['PercNonOrigSer'] = data['NonOrigSer'] / data['Servers']
# data['PercNonOrigObj'] = data['NumObjsNonOrigin'] / data['RetObjs']
# data.loc[~np.isfinite(data['PercNonOrigSer']), 'PercNonOrigSer'] = 0
# data.loc[~np.isfinite(data['PercNonOrigObj']), 'PercNonOrigObj'] = 0
# data['PercOrigObj'] = data['RetObjsOrig'] / data['RetObjs']
# data['SizePerObj'] = data['PageSize'] / data['RetObjs']
# data.loc[~np.isfinite(data['PercOrigObj']), 'PercOrigObj'] = 0
# data.loc[~np.isfinite(data['SizePerObj']), 'SizePerObj'] = 0

data['PageSize'] = np.log1p(data['PageSize'] + 0.000001)
data['JSSize'] = np.log1p(data['JSSize'] + 0.000001)
data['JSNum'] = np.log1p(data['JSNum'] + 0.000001)
data['ReqObjs'] = np.log1p(data['ReqObjs'] + 0.000001)
data['RetObjs'] = np.log1p(data['RetObjs'] + 0.000001)
data['Servers'] = np.log1p(data['Servers'] + 0.000001)
data['ImageNum'] = np.log1p(data['ImageNum'] + 0.000001)
# data['LogImageSize'] = np.log1p(data['ImageSize'] + 0.000001)
# data['LogOrigSer'] = np.log1p(data['OrigSer'] + 0.000001)
data['NonOrigSer'] = np.log1p(data['NonOrigSer'] + 0.000001)
data['HTMLNum'] = np.log1p(data['HTMLNum'] + 0.000001)
# data['LogCSSNum'] = np.log1p(data['CSSNum'] + 0.000001)
# data['LogPercNonOrigSer'] = np.log1p(data['PercNonOrigSer'] + 0.000001)
# data['LogPercNonOrigObj'] = np.log1p(data['PercNonOrigObj'] + 0.000001)
# data['LogPercOrigObj'] = np.log1p(data['PercOrigObj'] + 0.000001)

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

data["LoadTime"] = np.log1p(data["LoadTime"])

data_cols = data.columns
data.PageSize = data.PageSize / 10**6

feature_cols = data_cols.drop(["LoadTime", "VP", "AllMax1", "ImageSize", "OrigSer", "CSSNum"])
load_time = data['LoadTime']
features = data[feature_cols]

######## Feature skewness after log transformation ########
skewness = features.apply(lambda x: skew(x))
print "Features Skewness:\n", skewness

######## Plotting the histogram of the log-transformed features ########
features.hist()
plt.show()

#####*************** Feature Scaling and Handling Skewness **********************#####
# scaler = Normalizer()
# scaler = RobustScaler()
# scaler = StandardScaler()
# scaler = MinMaxScaler(feature_range=(0, 1))
# data[feature_cols] = np.log1p(data[feature_cols] + 0.000001)
# features[:] = scaler.fit_transform(features)
# data[feature_cols] = scaler.fit_transform(data[feature_cols])
# features = data[feature_cols]

#####*************** Feature Correlations **********************#####
###### Correlation Matrix ######
corrmat = data.corr()

###### Heatmap correlation plot ######
## number of variables for heatmap ##
k = 12
cols = corrmat.nlargest(k, 'LoadTime')['LoadTime'].index
cm = np.corrcoef(data[cols].values.T)
print "Correlations:\n", corrmat.nlargest(k, 'LoadTime')['LoadTime']
sns.set(font_scale=0.8)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#####*************** Scatter Plots **********************#####
fig, axes = plt.subplots(nrows=3, ncols=3)
data.plot(ax=axes[0,0], kind='scatter',x='ReqObjs', y='LoadTime')
data.plot(ax=axes[0,1], kind='scatter',x='RetObjs', y='LoadTime')
data.plot(ax=axes[0,2], kind='scatter',x='Servers', y='LoadTime')
data.plot(ax=axes[1,0], kind='scatter',x='NonOrigSer', y='LoadTime')
data.plot(ax=axes[1,1], kind='scatter',x='JSNum', y='LoadTime')
data.plot(ax=axes[1,2], kind='scatter',x='ImageNum', y='LoadTime')
data.plot(ax=axes[2,0], kind='scatter',x='HTMLNum', y='LoadTime')
# data.plot(ax=axes[2,1], kind='scatter',x='PercNonOrigObj', y='LoadTime')
# data.plot(ax=axes[2,2], kind='scatter',x='PercNonOrigSer', y='LoadTime')
plt.show()

#####*************** Principal Component Analysis **********************#####
pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(data.loc[:, feature_cols].values)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC-1', 'PC-2'])
principalDf.plot(kind='scatter', x='PC-1', y='PC-2')
plt.show()

principalDf['LoadTime'] = data.LoadTime.values
principalDf['LoadTime_bins'] = pd.cut(principalDf.loc[:, 'LoadTime'], [0, 10, 20, 50, 180], 
                                   labels = ['0-10', '10-20', '20-50', '>50'])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['0-10', '10-20', '20-50', '>50']
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['LoadTime_bins'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC-1']
               , principalDf.loc[indicesToKeep, 'PC-2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid('on')
plt.show()


#####*************** Regression Models **********************#####

################### Cross Validation ###################
def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model, features, load_time, scoring="neg_mean_squared_error", cv = 10))
	return(rmse)

def r2_score_cv(model):
	r2_score = cross_val_score(model, features, load_time, scoring="r2", cv = 10)
	return(r2_score)

################### Ridge Regression ###################
ridge_predicted = cross_val_predict(Ridge(alpha = 10), features, load_time, cv=10)
alphas = [0.0005, 0.001, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
cv_ridge_r2_score = [r2_score_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
# cv_ridge = pd.Series(cv_ridge, index = alphas)
# cv_ridge.plot(title = "Validation")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

######################### Kernel Ridge Regression #########################
# cv_kernelRidge = [rmse_cv(KernelRidge(alpha = alpha, kernel='polynomial', degree=2, coef0=2.5)).mean() for alpha in alphas]
# cv_kernelRidge_r2_score = [r2_score_cv(KernelRidge(alpha = alpha, kernel='polynomial', degree=2, coef0=2.5)).mean() for alpha in alphas]
# cv_kernelRidge = pd.Series(cv_kernelRidge, index = alphas)
# cv_kernelRidge.plot(title = "Validation")
# plt.xlabel("alpha")
# plt.ylabel("rmse")
# plt.show()

######################### Lasso Regression #########################
# model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(features, load_time)
# model_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]
min_lasso_error = 1000000
for alpha in alphas:
	lasso_error = rmse_cv(Lasso(alpha = alpha)).mean()
	if lasso_error < min_lasso_error:
		min_lasso_error = lasso_error
		min_alpha = alpha

best_lasso_model = Lasso(min_alpha).fit(features, load_time)
lasso_r2_scores = cross_val_score(best_lasso_model, features, load_time, scoring='r2', cv=10)
# coef = pd.Series(best_lasso_model.coef_, index = feature_cols)
# plt.rcParams['figure.figsize'] = (8.0, 10.0)
# coef.plot(kind = "barh")
# plt.title("Coefficients in the Lasso Model")


######################### ElasticNet Regression #########################
ElasticNet(random_state=0)
cv_elasticNet = [rmse_cv(ElasticNet(alpha = alpha, random_state=0)).mean() for alpha in alphas]
cv_elasticNet_r2_score = [r2_score_cv(ElasticNet(alpha = alpha, random_state=0)).mean() for alpha in alphas]
# cv_elasticNet = pd.Series(cv_elasticNet)


######################### XGBoost Regression #########################
# xgb1 = XGBRegressor(colsample_bytree=0.2,
#                  learning_rate=0.05,
#                  max_depth=3,
#                  n_estimators=1200
#                 )

# xgboost_cv_rmse = np.sqrt(-cross_val_score(xgb1, features, load_time, cv = 10, n_jobs=-1, scoring='neg_mean_squared_error'))
# xgb_test = XGBRegressor(learning_rate=0.05, n_estimators=500, max_depth=3, colsample_bytree=0.4)
# cv_score = cross_val_score(xgb_test, train_unskew.drop(['SalePrice','Id'], axis = 1), train_unskew['SalePrice'], cv = 5, n_jobs=-1)

###### Splitting Data into Train and Test data ######
# X_train = features[:-20]
# X_test = features[-20:]
# Y_train = load_time[:-20]
# Y_test = load_time[-20:]


######################### Log Prediction #########################
# def log_prediction(regr, features, load_time):
#        X_train, X_test, y_train, y_test = train_test_split(features, load_time, test_size=0.2)
#        y_train = np.log1p(y_train)
#        regr.fit(X_train, y_train)
#        y_pred = regr.predict(X_test)
#        y_pred_series = pd.Series(y_pred)
#        print y_pred_series.describe()
#        y_pred = np.exp(y_pred) - 1
#        y_pred_series = pd.Series(y_pred)
#        print y_pred_series.describe()
#        print "Root Mean Squared Error with log: ", sqrt(abs(mean_squared_error(y_test, y_pred)))
#        print "R2 Score with log: ", r2_score(y_test, y_pred)

# log_prediction(regr, features, load_time)
# log_prediction(regr, features, load_time)
# log_prediction(regr, features, load_time)


######################### Linear Regression #########################
regr = LinearRegression()
predicted = cross_val_predict(regr, features, load_time, cv=10)

mean_square_scores = cross_val_score(regr, features, load_time, scoring='neg_mean_squared_error', cv=10)
rms = sqrt(abs(mean_square_scores.mean()))
linear_reg_r2_scores = cross_val_score(regr, features, load_time, scoring='r2', cv=10)

X_train, X_test, y_train, y_test = train_test_split(features, load_time, test_size=0.2)
regr.fit(X_train, y_train)

######### Linear Regression Feature Importances #########
print "Linear Regression Features Coefficients: "
for coefficient in list(zip(feature_cols, regr.coef_)):
	print coefficient


######################### Random Forest Regressor #########################
rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
# rf_cv_mse = -cross_val_score(rf, features, load_time, scoring="neg_mean_squared_error", cv = 10)
rf_cv_r2_score = cross_val_score(rf, features, load_time, scoring="r2", cv = 5)
# rf.fit(X_train, y_train)

# predicted_train = rf.predict(X_train)
# predicted_test = rf.predict(X_test)
# test_score = r2_score(y_test, predicted_test)
# spearman = spearmanr(y_test, predicted_test)
# pearson = pearsonr(y_test, predicted_test)
# print "-----------------------------------------------------------------"
# print "Cross Validation RMSE: ", sqrt(rf_cv_mse.mean())
# print "Feature Importances: ", rf.feature_importances_
# print 'Out-of-bag R-2 score estimate: ', rf.oob_score_
# print 'Test data R-2 score: ', test_score
# print 'Test data Spearman correlation: ', spearman[0]
# print 'Test data Pearson correlation: ', pearson[0]


#####*************** Prediction Plot ***************#####
fig, ax = plt.subplots()
ax.scatter(load_time, ridge_predicted, edgecolors=(0, 0, 0))
ax.plot([load_time.min(), load_time.max()], [load_time.min(), load_time.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# residuals = pd.Series(abs(load_time - ridge_predicted))
# residuals = pd.Series(sqrt(abs(mean_squared_error(load_time, ridge_predicted))))
# data['residuals'] = residuals
# residuals.where(residuals > 25).hist()
# plt.show()
# high_residuals = data.loc[(data['residuals'] > 25), ['LoadTime', 'PageSize', 'ReqObjs', 'RetObjs', 'Servers', 'VP', 'residuals', 'AllMax1']]
# print high_residuals.sort('residuals', ascending=False).head(50).to_string()
# print high_residuals.describe().to_string()


######### Printing the results #########
load_time_range = data['LoadTime'].max() - data['LoadTime'].min()
print "-----------------------------------------------------------------"
print "Linear Regression Load Time Prediction: ", predicted.mean()
print "Linear Regression Root Mean Square Error: ", rms
print "Linear Regression Normalized Root Mean Square Error: ", float(rms) / load_time_range
print "Linear Regression R2 Score: ", linear_reg_r2_scores.mean()
# print "Linear Regression Score: ", regr.score(X_test, y_test)

# print "***************************************************"
# print "XGBoost Root Mean Square Error: ", xgboost_cv_rmse.mean()
# print "***************************************************"
# print "KernelRidge Root Mean Square Error: ", cv_kernelRidge.min()
# print "KernelRidge Normalized Root Mean Square Error: ", cv_kernelRidge.min() / load_time_range
# print "KernelRidge R2 Score: ", max(cv_kernelRidge_r2_score)
print "***************************************************"
print "Random Forest Regressor Cross Validation R-2 Score: ", rf_cv_r2_score.mean()
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







