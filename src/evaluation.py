#visualizamos los valores actuales con los valores de la predicción 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, linear_regression_predictions, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: valores actuales versus valores de la predict")
plt.show()
#utilizamos otro método de predict con Decision Tree 
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_train, y_train)
decision_tree_predictions = decision_tree_regressor.predict(X_test)
#calculamos las métricas de la Decision Tree Regressor 
mse_tree = mean_squared_error(y_test, decision_tree_predictions)
r2_tree = r2_score(y_test, decision_tree_predictions)
print("Decision Tree Regressor - Mean Squared Error: {:.2f}".format(mse_tree))
print("Decision Tree Regressor - R^2 Score: {:.2f}".format(r2_tree))
#visualización valorers actuales versus valores de la decision Tree predict 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, decision_tree_predictions, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree Regressor: valores actuales versus valores de la Decision Tree predict")
plt.show()
#tercer modelo de predict con Random Forest
random_forest_regressor = RandomForestRegressor(n_estimators=100)
random_forest_regressor.fit(X_train, y_train)
random_forest_predictions = random_forest_regressor.predict(X_test)
#calculamos las métricas de Random Forest 
mse_forest = mean_squared_error(y_test, random_forest_predictions)
r2_forest = r2_score(y_test, random_forest_predictions)
print("Random Forest Regressor - Mean Squared Error: {:.2f}".format(mse_forest))
print("Random Forest Regressor - R^2 Score: {:.2f}".format(r2_forest))
plt.figure(figsize=(10, 6))
plt.scatter(y_test, random_forest_predictions, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regressor: datos actuales versus datos del Random Forest predict")
plt.show()
#podemos probar con el método GridSearchCV 
#establecemos nuevas variables:
param_grid_lr = {}
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3]
}
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
#creamos los modelos 
models = {
    'Linear Regression': (LinearRegression(), param_grid_lr),
    'Decision Tree Regressor': (DecisionTreeRegressor(), param_grid_dt),
    'Random Forest Regressor': (RandomForestRegressor(), param_grid_rf),
    'K-Neighbors Regressor': (KNeighborsRegressor(), param_grid_knn),
    'Support Vector Regressor': (SVR(), param_grid_svr),
    'Gradient Boosting Regressor': (GradientBoostingRegressor(), param_grid_gb)
}
#entrenamos los modelos para elegir el mejor modelo
best_models = {}
for model_name, (model, param_grid) in models.items():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"{model_name} - Best Parameters: {grid_search.best_params_}")
    print(f"{model_name} - Best CV Score: {grid_search.best_score_:.2f}")
    #probamos el mejor modelo
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"{model_name} - Test Mean Squared Error: {mse:.2f}")
    print(f"{model_name} - Test R^2 Score: {r2:.2f}")
    plot_predictions(y_test, predictions, model_name)
    #visualización de los valores actuales y del modelo predict 
    def plot_predictions(y_test, predictions, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.3)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linestyle='--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f'{model_name}: Actual vs Predicted Values')
    plt.show()
    for model_name, best_model in best_models.items():
    evaluate_model(best_model, X_test, y_test, model_name)
