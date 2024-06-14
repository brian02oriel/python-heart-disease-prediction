import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class ModelPipeline:
    def __init__(
        self, 
        predict) -> None:
        self.predict = predict
        pass

    def get_features(self, df):
        bins = np.arange(20, 90, 10)
        labels = ['2030', '3040', '4050', '5060', '6070', '7080']
        df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        return df


    def execute(self):
        df = pd.DataFrame(data=[self.predict], columns=[
            "age",
            "sex_",
            "chest_pain_type",
            "resting_blood_pressure",
            "serum_cholestoral",
            "fasting_blood_sugar",
            "resting_electrocardiographic_results",
            "max_heart_rate",
            "exercise_induced_angina",
            "oldpeak",
            "ST_segment",
            "major_vessels",
            "thal"])
        
        
        numerical_features = ['age', 'resting_blood_pressure', 'serum_cholestoral', 'max_heart_rate', 'oldpeak', 'ST_segment', 'major_vessels']
        categorical_features = ['sex_', 'age_category', 'chest_pain_type', 'fasting_blood_sugar', 'resting_electrocardiographic_results', 'exercise_induced_angina', 'thal']

        for col in numerical_features:
            df[col] = df[col].astype(float)
        df = self.get_features(df)

    