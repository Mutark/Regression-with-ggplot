# Regression-with-ggplot
#For young data analyst, Regression is an important basics to learn using R or Python with some additional analytical visuals need to explain the results. 
#Simple Regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy import stats

# Read CSV file
h2 = pd.read_csv("C:/Users/HP/OneDrive - Western Illinois University/Desktop/Lectures/2nd Semester/DS435 data mining/Regression/d3.csv")

# Remove missing values
h1 = h2.dropna()

# Scatter plot of WT and SUGAR
plt.scatter(h1['AGE'], h1['BMI'])
plt.xlabel('AGE')
plt.ylabel('BMI')
plt.show()

# Model fitting
d1 = sm.formula.ols('BMI ~ AGE', data=h1).fit()

# Model summary
print(d1.summary())

# Model prediction
AGE_test = pd.DataFrame({'AGE': [64,48,88,52,90]})
prediction = d1.predict(AGE_test)

# Scatter plot with fitted regression line
plt.scatter(h1['AGE'], h1['BMI'])
plt.plot(h1['AGE'], d1.predict(h1['AGE']), color='red')
plt.xlabel('AGE')
plt.ylabel('BMI')
plt.show()

# Scatter plot with fitted regression line using ggplot
!pip install ggplot
from ggplot import *
ggplot(h1, aes(x='AGE', y='BMI')) + \
    geom_point(size = 3, color = "firebrick") + \
    geom_smooth(method = "lm", se = False, color = "black") + \
    labs(x = "Age of Respondent", y = "Body Mass Index") + \
    theme_classic()

# Confidence interval of the coefficient
print(d1.conf_int())

# Prediction
yhat = d1.get_prediction(pd.DataFrame({'AGE': [93]})).summary_frame()
print(yhat)

# 95% confidence band
sns.lmplot(x='AGE', y='BMI', data=h1, ci=95)

# 95% prediction interval
pred = d1.get_prediction(h1).summary_frame(alpha=0.05)
h2 = pd.concat([h1, pred], axis=1)
plt.fill_between(h2['AGE'], h2['mean_ci_lower'], h2['mean_ci_upper'], color='black', alpha=.2)
plt.plot(h1['AGE'], d1.predict(h1['AGE']), color='black')
plt.scatter(h1['AGE'], h1['BMI'], color='firebrick')
plt.xlabel('AGE')
plt.ylabel('BMI')
plt.show()

# Residuals
residual = pd.DataFrame(d1.resid)
residual.columns = ['Residual']

# Shapiro-Wilk test
print(stats.shapiro(residual['Residual']))

# Boxplot and histogram
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.boxplot(y=residual['Residual'], ax=ax1)
sns.histplot(residual['Residual'], kde=True, ax=ax2)
sns.kdeplot(residual['Residual'], ax=ax2, color='r')
plt.show()

#Multiple Regression
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read data
h2 = pd.read_csv("C:/Users/HP/OneDrive - Western Illinois University/Desktop/Lectures/2nd Semester/DS435 data mining/Regression/d3.csv")

# Remove missing values
h1 = h2.dropna()

# Fit multiple regression model
d2 = sm.OLS(h1['BMI'], sm.add_constant(h1[['AGE', 'WT', 'LDL', 'TC', 'HT']])).fit()
print(d2.summary())

#Drop and Add new variable
d3 = sm.OLS(h1['BMI'], sm.add_constant(h1[['WT', 'LDL', 'TC', 'HT', 'INCOME', 'SBP']])).fit()
print(d3.summary())

# Calculate residuals and perform Shapiro-Wilk test for normality
d3 = pd.DataFrame({'Residual': d2.resid})
sw_pvalue = stats.shapiro(d3['Residual'])[1]
print(f"Shapiro-Wilk test p-value: {sw_pvalue:.4f}")

# Create boxplot and histogram of residuals
fig, ax = plt.subplots(ncols=2, figsize=(10,4))
ax[0].boxplot(d3['Residual'])
ax[0].set_title('Boxplot of Residuals')
ax[1].hist(d3['Residual'], density=True)
ax[1].set_title('Histogram of Residuals with Normal Curve')
x = np.linspace(min(d3['Residual']), max(d3['Residual']), 100)
f = stats.norm.pdf(x, loc=d3['Residual'].mean(), scale=d3['Residual'].std())
ax[1].plot(x, f, color='red', lw=2)
plt.show()

# Create QQ plot of residuals
fig, ax = plt.subplots()
sm.qqplot(d3['Residual'], ax=ax, line='s')
ax.set_title('QQ Plot of Residuals')
plt.show()
