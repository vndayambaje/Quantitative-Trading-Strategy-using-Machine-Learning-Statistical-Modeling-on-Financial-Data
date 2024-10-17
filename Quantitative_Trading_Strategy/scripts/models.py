import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Train RandomForest Classifier with hyperparameter tuning
def train_random_forest(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(random_state=42)
    
    # Perform hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Get the best model after tuning
    best_rf = grid_search.best_estimator_
    
    # Validate the model on the validation set
    y_val_pred = best_rf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred)
    
    print("Best RandomForest Hyperparameters:", grid_search.best_params_)
    
    return best_rf, val_accuracy, val_report

# Train XGBoost Classifier with hyperparameter tuning
def train_xgboost(X_train, y_train, X_val, y_val):
    xgb = XGBClassifier(random_state=42, use_label_encoder=False)
    
    # Perform hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_xgb = grid_search.best_estimator_
    
    # Validate the model
    y_val_pred = best_xgb.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred)
    
    print("Best XGBoost Hyperparameters:", grid_search.best_params_)
    
    return best_xgb, val_accuracy, val_report

# Train LSTM model with validation
def train_lstm_model(X_train, y_train, X_val, y_val):
    # Reshape X data for LSTM model (LSTM expects 3D input: [samples, timesteps, features])
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_data=(X_val_reshaped, y_val))
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val_reshaped)
    y_val_pred_reshaped = (y_val_pred > 0.5).astype(int).flatten()

    # Calculate accuracy and classification report
    val_accuracy = accuracy_score(y_val, y_val_pred_reshaped)
    val_report = classification_report(y_val, y_val_pred_reshaped, zero_division=1)

    return model, val_accuracy, val_report

# StandardScaler for feature scaling
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled