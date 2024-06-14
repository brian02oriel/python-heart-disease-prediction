import pandas as pd
import numpy as np
import joblib

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
        # TODO: save the feature columns for the prediction
        
        features = ['age', 'resting_blood_pressure', 'serum_cholestoral', 'max_heart_rate',
       'oldpeak', 'ST_segment', 'major_vessels', 'sex__0.0', 'sex__1.0',
       'age_category_2030.0', 'age_category_3040.0', 'age_category_4050.0',
       'age_category_5060.0', 'age_category_6070.0', 'age_category_7080.0',
       'chest_pain_type_1.0', 'chest_pain_type_2.0', 'chest_pain_type_3.0',
       'chest_pain_type_4.0', 'fasting_blood_sugar_0.0',
       'fasting_blood_sugar_1.0', 'resting_electrocardiographic_results_0.0',
       'resting_electrocardiographic_results_1.0',
       'resting_electrocardiographic_results_2.0',
       'exercise_induced_angina_0.0', 'exercise_induced_angina_1.0',
       'thal_3.0', 'thal_6.0', 'thal_7.0']
        for col in numerical_features:
            df[col] = df[col].astype(float)
        df = self.get_features(df)

        pipeline = joblib.load('pipelines/pipeline.joblib')
        print(pipeline.feature_names_in_)
        df = pd.DataFrame(data=pipeline.transform(df), columns=features)
        print(df)
        return df

    