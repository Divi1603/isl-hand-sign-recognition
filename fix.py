import pandas as pd
df = pd.read_csv('live_landmarks.csv')
df['label'] = df['label'].str.strip("[]'")
print(df['label'].value_counts())
df.to_csv('live_landmarks.csv', index=False)
print('Done!')