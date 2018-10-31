# Instacart_supervisedLearning

Problem Statement: To predict the repurchase of items of customer in Instacart.

Data Source: Instacart Kaggle Dataset

Datasets brief description:
* Aisles: Details about the aisle and its descriptions
* Departments: Details about the department and its descriptions
* order_products_prior: These files specify which products were purchased in each order. order_products__prior.csv contains previous order contents for all customers. 
* orders: This file tells to which set (prior, train, test) an order belongs. You are predicting reordered items only for the test set orders. 'order_dow' is the day of week.
* products: Details about the product, description, its aisle id and department.

Tools used:
* Data Visulization: matplotlib,seaborn
* Tools used: Pandas, sklearn, SQL

Algorithms:
* Logistic Regression
* Decision Tree

Approach:
* Step 1: Exploratory Data Analysis on the dataset
* Step 2: Feature engineering 
* Step 3: Class imbalance checked and appropriate weights are assigned.
* Step 4: Applying algorithms and checked for the F1 score
* Step 5: Based on the F1 score, precision Logistic regression performs better than DecisionTree.
