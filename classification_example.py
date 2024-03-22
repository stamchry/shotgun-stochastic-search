from main.generate_test_df import generate_test_df_binary_regression
from main.sss import SSS
import seaborn as sns
import matplotlib.pyplot as plt

# Generate simulated data for binary regression
df = generate_test_df_binary_regression(50, 20, 2, 1)

# Extract features (X) and target (y)
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# Initialize and fit the SSS model
sss_instance = SSS(iterations=1000, hyperparameter=2/20, tau=1, regression_type='binary')
result = sss_instance.fit(X, y, df, num_of_best_scores=1000)

# Drop variables that did not appear in the search
result = result[result != 0].dropna(axis=1, how='all').fillna(0)

# Calculate and print relative importance of each variable
for i in result.columns[:-2]:
    relative_importance = result[result[i] == 1]['Relative Importance'].sum()
    print(f'Relative importance of variable {i}: {relative_importance}')

# Plot pairplot for variables discovered by SSS
sns.pairplot(df[result.columns.tolist()[:-2] + ['target']], hue='target')
plt.show()
