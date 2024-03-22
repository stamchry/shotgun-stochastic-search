from main.generate_test_df import generate_test_df_linear_regression
from main.sss import SSS

# Iterate over the number of features for the experiment
for p in [10, 50, 100, 150, 200, 1000]:

    # Generate synthetic data
    df, X, y, coeff = generate_test_df_linear_regression(50, p, 4)

    # Create an object of the class SSS
    sss_instance = SSS(iterations=1000, hyperparameter=4/p, tau=1, delta=3, regression_type='linear')

    # Fit the data to the object
    result = sss_instance.fit(X, y, df, num_of_best_scores=1000)

    # Drop variables that did not appear in the search
    result = result[result != 0].dropna(axis=1, how='all').fillna(0)

    # Print the summary of the search
    print(f'SSS for {p} features')
    print('=======================================================================')

    for i in result.columns[:-2]:
        a = result[result[i] == 1]['Relative Importance'].sum()
        print(f'Relative importance of variable {i} : {a}')
    
    print('=======================================================================')
