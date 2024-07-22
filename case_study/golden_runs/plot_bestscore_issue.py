import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv("allruns.csv")

# Convert 'score' column to numeric
df['score'] = pd.to_numeric(df['score'])

df = df.drop(columns=['individual','patch','step_duration'])

# Correct the grouping to use a list for column subsetting
grouped_df = df.groupby(['model', 'issue', 'epoch'])[['model', 'issue', 'epoch', 'step', 'score']].apply(lambda x: x.loc[x['score'].idxmax()]).reset_index(drop=True)

# Pivot table to reshape data for plotting: index will be (model, issue), columns will be epoch, values will be score
pivot_df_scores = grouped_df.pivot_table(index=['model', 'issue'], columns='epoch', values='score', aggfunc='first')

# Pivot table for steps to use in annotations
pivot_df_steps = grouped_df.pivot_table(index=['model', 'issue'], columns='epoch', values='step', aggfunc='first')

# Set up the plotting environment for colorblind compatibility
sns.set(style="whitegrid", palette="colorblind")

# Create a heatmap of the scores
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(pivot_df_scores, annot=True, fmt=".2f", cmap="viridis", linewidths=.5)

""" # Add annotations for the number of steps on the heatmap
for (i, j), score in np.ndenumerate(pivot_df_scores):
    steps = pivot_df_steps.iat[i, j]
    text = f'\n({steps} steps)'
    plt.text(j + 0.5, i + 0.6, text, ha='center', va='center', color='grey', fontsize=8) """

plt.title('Score per Issue per Model Over Epochs')
plt.ylabel('Model, Issue')
plt.xlabel('Epoch')
plt.show()

# Filter the DataFrame to include only entries where the score is exactly 1.0
df2 = df

# Extract all data for epoch 0
epoch_zero_df = df[df['epoch'] == 0]

# Filter to include scores of 1.0 for epochs other than 0
high_score_df = df[(df['score'] == 1.0) & (df['epoch'] != 0)]

# Combine epoch 0 data with high-score data from other epochs
combined_df = pd.concat([epoch_zero_df, high_score_df]).drop_duplicates()

# Pivot data for heatmap creation: Models on one axis, Epochs x Issues on the other
heatmap_data = combined_df.pivot_table(values='step', index=['model'], columns=['epoch', 'issue'], aggfunc='min')

# Create a figure with subplots - one for each model
models = df['model'].unique()
fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(20, 6))

for ax, model in zip(axes, models):
    # Extract data for the current model
    model_data = heatmap_data.loc[model]

    # Ensure data is in a DataFrame format and check for dimension issues
    if isinstance(model_data, pd.Series):  # This means there is only one row or one column
        model_data = model_data.to_frame().T  # Convert to a DataFrame and transpose if necessary

    sns.heatmap(model_data, ax=ax, annot=True, cmap="viridis", fmt=".0f", linewidths=.5, linecolor='grey')
    
    ax.set_title(f'Steps to Score 1.0 for {model} (Including Baseline Epoch 0)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Issue')
    ax.invert_yaxis()  # Inverts the y-axis to align the heatmap with traditional matrix configurations

plt.tight_layout()
plt.suptitle('Comparative Number of Steps to Reach Score of 1.0 by Issue and Model Over Epochs (Including Baseline)', fontsize=16)
plt.show()