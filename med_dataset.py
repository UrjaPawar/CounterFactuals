from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

class Data:
    heart_path = "data/uci_heart.csv"
    cerv_path = "data/risk_factors_cervical_cancer.csv"
    diab_path = "data/diabetes.csv"

    def __init__(self, dataset_name, one_hot_encode= True, test_split = 0.2, shuffle_while_training = True, random_state = 24):
        self.df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.features = []
        self.target = []
        self.categorical = []
        self.one_hot = []
        self.continuous = []
        self.binary = []
        if dataset_name == "Heart DB":
            self.load_heart()
        elif dataset_name == "Cervical DB":
            self.load_cervical()
        elif dataset_name == "Diabetes DB":
            self.load_diabetes()
        if one_hot_encode:
            self.one_hot_encode()
        self.split_data(test_split, shuffle_while_training, random_state)

    def remove_missing_vals(self, filepath, missing_arr):
        print("Missing values will be removed")
        df = pd.read_csv(filepath, na_values=missing_arr)
        df = df.dropna()
        return df

    def split_data(self, test_split, shuffle, random_state):
        print("Splitting into training and testing sets")
        # explore the stratify option
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(self.target, axis=1), self.df[self.target], test_size=test_split,
                                                            random_state=random_state, shuffle = shuffle)
        self.train_df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)


    def load_heart(self):
        missing_values = ["?"]
        data = self.remove_missing_vals(self.heart_path, missing_values)

        data['Typical_Angina'] = (data['Chest_Pain'] == 1).astype(int)
        data['Atypical_Angina'] = (data['Chest_Pain'] == 2).astype(int)
        data['Asymptomatic_Angina'] = (data['Chest_Pain'] == 4).astype(int)
        data['Non_Anginal_Pain'] = (data['Chest_Pain'] == 3).astype(int)
        data = data.drop(columns=['Chest_Pain'])
        data["MHR"] = 220 - data["Age"]
        data["mhr_exceeded"] = data["MHR"] < data["MAX_Heart_Rate"]
        data["mhr_exceeded"] = data["mhr_exceeded"].astype(int)
        data = data.drop(columns=["MAX_Heart_Rate", "MHR"])
        self.features = ['Age', 'Sex', 'Typical_Angina', 'Atypical_Angina', 'Resting_Blood_Pressure',
                           'Fasting_Blood_Sugar', 'Rest_ECG', 'Colestrol', 'Asymptomatic_Angina', 'Non_Anginal_Pain',
                           'Slope', 'ST_Depression', 'Exercised_Induced_Angina', 'mhr_exceeded', 'Major_Vessels',
                           'Thalessemia']

        TARGET_COLUMNS = ['Target']
        data[TARGET_COLUMNS] = data[TARGET_COLUMNS] != 0
        data[TARGET_COLUMNS] = data[TARGET_COLUMNS].astype(int)
        self.continuous = ['Age', 'Colestrol', 'Resting_Blood_Pressure','ST_Depression']
        self.target = TARGET_COLUMNS[0]
        self.df = data[self.features + [self.target]]
        self.binary = ['Sex', 'Typical_Angina', 'Atypical_Angina', 'Fasting_Blood_Sugar', 'Asymptomatic_Angina',
                       'Non_Anginal_Pain',
                       'Exercised_Induced_Angina', 'mhr_exceeded']
        self.categorical = [feature for feature in self.features if feature not in self.continuous and feature not in self.binary]


    def load_diabetes(self):
        df = pd.read_csv(self.diab_path)
        self.df = df.drop(columns=['Insulin'])
        self.features = list(df.columns)
        self.features.remove('Outcome')
        self.target = "Outcome"
        self.continuous = self.features.copy()


    def load_cervical(self):
        df = pd.read_csv(self.cerv_path, na_values=["?"])
        numerical_s = ['Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies',
                     'Hormonal Contraceptives (years)',
                     'IUD (years)', 'STDs (number)', 'STDs: Number of diagnosis', 'Smokes (packs/year)',
                     'Smokes (years)']

        df = df.drop(
        columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 'Hormonal Contraceptives',
                 'IUD', 'Smokes', 'STDs', 'STDs:HIV',
                 'STDs:condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:genital herpes'])
        df = df.dropna()
        tar = 'Biopsy'
        feats = list(df.columns)
        feats.remove(tar)

        self.continuous = [feat for feat in feats if feat in numerical_s]
        self.categorical = df[feats].columns.difference(self.continuous)

        ada = ADASYN(random_state=28)
        X_ada, y_ada = ada.fit_resample(df[feats], df[tar])
        m2a = pd.DataFrame(X_ada)
        m2b = pd.DataFrame(y_ada)
        m2b.columns = ['Biopsy']
        self.df = m2a.join(m2b)
        self.target = 'Biopsy'
        self.features = list(self.df.columns)
        self.features.remove(self.target)

    def one_hot_encode(self):
        self.df = pd.get_dummies(self.df, columns=self.categorical)



