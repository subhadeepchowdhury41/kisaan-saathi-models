import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier

import pickle as pk

df = pd.read_csv('./Crop_recommendation.csv')

df['label'] = df['label'].map({'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3,
                               'pigeonpeas': 4, 'mothbeans': 5, 'mungbean': 6,
                               'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
                               'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13,
                               'muskmelon': 14, 'apple': 15, 'orange': 16, 'papaya': 17,
                               'coconut': 18, 'cotton': 19, 'jute': 20, 'coffee': 21})

X_train, X_test, Y_train, Y_test = train_test_split(df[['N', 'P', 'K', 'temperature',
                                                       'humidity', 'ph', 'rainfall']],
                                                    df['label'], random_state=42)

model = RandomForestClassifier(max_depth=12, random_state=42)
model.fit(X_train, Y_train)

pk.dump(model, open("model.pk", 'wb'))
