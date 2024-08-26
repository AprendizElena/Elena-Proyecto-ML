#establecemos las variables de entrenemiento 
#Features: 
X = cod_df.drop('price', axis=1)
#Target:
y = cod_df['price']
#realizamos el primer test de entrenamiento 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#verificamos el shape de los modelos de etrenamiento
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
#Predicciones usando la regresión 
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)
linear_regression_predictions = linear_regression_model.predict(X_test)
#calculamos las métricas de la regresión linear 
mse_linear = mean_squared_error(y_test, linear_regression_predictions)
r2_linear = r2_score(y_test, linear_regression_predictions)
print("Linear Regression - Mean Squared Error: {:.2f}".format(mse_linear))
print("Linear Regression - R^2 Score: {:.2f}".format(r2_linear))
