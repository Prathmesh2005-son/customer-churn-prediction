import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 5000

data = {
    'customer_id': range(1, n_samples + 1),
    'tenure': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'total_charges': np.random.uniform(100, 8000, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'partner': np.random.choice(['Yes', 'No'], n_samples),
    'dependents': np.random.choice(['Yes', 'No'], n_samples),
    'phone_service': np.random.choice(['Yes', 'No'], n_samples),
    'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
    'payment_method': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples
    ),
}

df = pd.DataFrame(data)

churn_probability = []

for _, row in df.iterrows():
    prob = 0.2

    if row['tenure'] < 12:
        prob += 0.3

    if row['contract'] == 'Month-to-month':
        prob += 0.3

    if row['internet_service'] == 'Fiber optic':
        prob += 0.15

    if row['online_security'] == 'No' and row['internet_service'] != 'No':
        prob += 0.1

    if row['monthly_charges'] > 80:
        prob += 0.1

    if row['senior_citizen'] == 1:
        prob += 0.05

    churn_probability.append(min(prob, 0.95))

df['churn'] = np.random.binomial(1, churn_probability)

import os
os.makedirs('data', exist_ok=True)

df.to_csv('data/churn_data.csv', index=False)

print(f"Dataset created: {len(df)} samples")
print(f"Churned customers: {df['churn'].sum()} ({df['churn'].mean():.1%} churn rate)")