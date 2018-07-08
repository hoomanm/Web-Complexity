# A Longitudinal Assessment of Website Complexity

Web Crawler:
  - Automatically collects HAR (HTTP Archive) files from our set of target websites using Selenium WebDriver in Java and HAR       Export Trigger Firefox add-on.

HAR File Parser:
  - A parser for extracting the desired infromation from the collected HAR files using the haralyzer module in Python.
  
Regression Models:
 - Several regression models such as: Linear Regression, Ridge and Lasso Regression, and Random Forest Regressor for              predicting the page load times using different features like: number and size of the objects, number of the contacted          servers, etc.
