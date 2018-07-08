# Web-Complexity
A Longitudinal Assessment of Website Complexity

Web Crawler:
- Automatically collects HAR (HTTP Archive) files from our set of target websites using Selenium WebDriver in Java and HAR Export Trigger Firefox add-on.
 
 Har File Parser:
  - A parser for extracting the desired infromation from the collected HAR files using the haralyzer module in Python.
  
Regression Models:
 - Set of regression models for predicting the page load times using different features extracted from HAR files such as: number of objects, size of objects, number of contacted servers, etc.
