from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd

def predict_country_of_origin(make_input, model_input):
    if os.path.exists('dataset_l.xlsx'):
        df = pd.read_excel('dataset_l.xlsx')
        # Check if required columns are present
        if 'Make' in df.columns and 'Model' in df.columns and 'Country of Origin' in df.columns:
            # Drop duplicates
            df = df.drop_duplicates()
            # Prepare features and labels
            X = df[['Make', 'Model']]
            y = df['Country of Origin']
            
            # Handle missing values
            X = X.dropna()
            y = y[X.index]
            
            # Encode features
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
            X_encoded = one_hot_encoder.fit_transform(X)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            
            # Train the model
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_encoded, y_encoded)
            
            # Prepare input data
            input_data = [[make_input, model_input]]
            input_encoded = one_hot_encoder.transform(input_data)
            
            # Predict
            country_pred = rf_classifier.predict(input_encoded)
            # Decode the predicted label
            country_of_origin = label_encoder.inverse_transform(country_pred)
            return country_of_origin[0]
        else:
            return None
    else:
        return None