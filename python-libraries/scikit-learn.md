## scikit-learn Library Notes

### Overview
- **scikit-learn** is an open-source machine learning library for Python.
- It features various classification, regression, and clustering algorithms.
- Built on NumPy, SciPy, and Matplotlib.

### Installation
```sh
pip install scikit-learn
```

### Key Features
1. **Supervised Learning Algorithms**:
    - Classification: K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Support Vector Machines (SVM), etc.
    - Regression: Linear Regression, Ridge Regression, Lasso, Elastic Net, etc.

2. **Unsupervised Learning Algorithms**:
    - Clustering: K-Means, DBSCAN, Hierarchical Clustering, etc.
    - Dimensionality Reduction: Principal Component Analysis (PCA), Independent Component Analysis (ICA), etc.

3. **Model Selection**:
    - Cross-validation: `cross_val_score`
    - Grid Search: `GridSearchCV`
    - Randomized Search: `RandomizedSearchCV`
    
4. **Preprocessing**:
    - Standardization: `StandardScaler`
    - Normalization: `MinMaxScaler`
    - Encoding: `OneHotEncoder`, `LabelEncoder`

### Basic Workflow
1. **Importing Libraries**:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    ```

2. **Loading Data**:
    ```python
    import pandas as pd

    data = pd.read_csv('path_to_dataset.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    ```

3. **Splitting Data**:
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

4. **Training Model**:
    ```python
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    ```

5. **Making Predictions**:
    ```python
    y_pred = model.predict(X_test)
    ```

6. **Evaluating Model**:
    ```python
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    ```

### Useful Tips
- **Cross-Validation**:
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5)
    print(f'Cross-Validation Scores: {scores}')
    ```

- **Hyperparameter Tuning**:
    ```python
    from sklearn.model_selection import GridSearchCV

    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f'Best Parameters: {grid_search.best_params_}')
    ```

- **Handling Imbalanced Data**:
    - Use techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.
    ```python
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    ```

### Resources
- **Documentation**: [scikit-learn](https://scikit-learn.org/stable/)
- **Cheat Sheet**: [Scikit-Learn Cheat Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)
- **Examples**: Check out the [Examples Gallery](https://scikit-learn.org/stable/auto_examples/index.html) for various use cases.
