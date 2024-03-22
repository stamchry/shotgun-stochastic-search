import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from main.sss import SSS  # Import your SSS class from main.sss

# Import Data
df = pd.read_csv('tecator_data.csv')

# Center and Normalize the data
df_centered = df - df.mean()
df_normalized = df_centered / df.std()

X = df_normalized.iloc[:, :-1].to_numpy()  # Features
y = df_normalized.iloc[:, -1].to_numpy()   # Target

# Define a list with the values of the hyperparameter to be used for the experiment
hyperparameter_values = [5 / 100, 10 / 100, 20 / 100, 40 / 100]

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Iterate over the different values of the hyperparameter
for idx, hyperparameter in enumerate(hyperparameter_values):

    # Create an instance of the SSS class
    sss_instance = SSS(iterations=10, hyperparameter=hyperparameter, tau=1, delta=3, regression_type='linear')

    # Fit the data
    result = sss_instance.fit(X, y, df_normalized, num_of_best_scores=1000)

    # Summarize the output for the subplots
    rel_imp_df = pd.DataFrame(index=result.columns[:-2], columns=['Relative Importance', 'Index of Covariate'])
    for j, col in enumerate(result.columns[:-2]):
        rel_imp_df['Relative Importance'][col] = result[result[col] == 1]['Relative Importance'].sum()
        rel_imp_df['Index of Covariate'][col] = j + 1

    row = idx // 2
    col = idx % 2

    # Create scatter plot for relative importance
    sns.scatterplot(data=rel_imp_df, x='Index of Covariate', y='Relative Importance', ax=axs[row, col])

    # Set title for the subplot
    axs[row, col].set_title(f'π = {int(hyperparameter * 100)}/100')

    # Set x-axis ticks to show every index of covariate
    axs[row, col].set_xticks(rel_imp_df['Index of Covariate'].astype(int))

    # Hide x-axis labels
    axs[row, col].tick_params(axis='x', labelbottom=False, bottom=True)

# Show the plot
plt.tight_layout()
plt.show()
