## Regression model and Flask REST API

Project structure

```bash
ml-price-predict-app
|
|-- data/
|   |-- avocado.csv
|
|-- flask_app/
|   |-- sql_code/
|      |-- create_table.sql
|   |-- templates/
|      |-- index.html
|      |-- recipes.html
|   |-- screenshots/
|      |-- flask_gui.png
|   |-- app.py
|   |-- db.yaml
|
|-- plots/
|   |-- feature_betas.png
|   |-- histogram_pred_y_test.png
|   |-- regression_gradient_descent_mse.png
|
|-- README.md
|-- analysis_output.txt
|-- .gitignore
|-- linear_regression.py
|-- price_prediction_model.py
|-- requirements.txt
```

### Comments about price_prediction_model to predict price of avocados
Current functionality
* Applies multivariate regression using gradient descent minimisation
* Applies train, test split based on random choice (although this may vary if dataset is updated)
* Basic EDA plots
* Calculates MSE, standard errors and p-values of the relevant features using only continuous features

Potential improvements
* Function to check multicollinearity, auto-correlation, Homoscedasticity and normality of the features before fitting.
* Some advanced EDA plots like correlation matrices, scatter and distribution plots
* Function to allow one hot encoding of categorical variables
* Function to apply L1 or L2 regularization penalty to minimise overfitting, and select best features before fitting.
* Principal Component Analysis for dimensionality reduction, although since there weren't many features here, it can be ignored.
* k-fold cross validation to get an average MSE
* Function for selecting the best hyperparameters for the Linear Regression model and minimise MSE
* Unit tests have not been implemented, hence that could be looked into as well.

### Comments about the Flask CRUD app for recipes
Current functionality
* Flask REST API functionality with MySQL database incorporating mainly GET and POST methods
* Allows user to create, update, delete and search for recipes based on recipe name and ingredients

Potential improvements
* Prevent empty records from being entered into the database
* Include user authentication so that only particular users or admin can access all recipes at the /recipes endpoint
* Currently vulnerable to SQL injection hence that could be improved on

![Flask Recipe App](https://github.com/amj18/ml-price-predict-app/blob/master/flask_app/screenshots/flask_gui.PNG)
