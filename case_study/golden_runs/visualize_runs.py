import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('allruns.csv')

# Group the data by some value (e.g., 'category')
#grouped_df = df.groupby('model')

""" numeric_cols = df.select_dtypes(include='number').columns.tolist()
id_vars = ['model', 'issue']
value_vars = [col for col in numeric_cols if col not in id_vars]

# Melt the DataFrame to a long format with only numeric columns
melted_data = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Metric', value_name='Value')

# Set the size of the plot
plt.figure(figsize=(14, 8))

# Create a grouped bar plot
g = sns.catplot(
    data=melted_data,
    kind="bar",
    x="Metric",
    y="Value",
    hue="issue",
    col="model",
    ci=None,
    palette="viridis",
    height=6,
    aspect=1
)

# Add title and adjust the layout
g.fig.suptitle('Comparison of Metrics by Issue Type for Each Model', y=1.02)
g.set_axis_labels("Metrics", "Values")
g._legend.set_title("Issue Type")

# Show the plot
plt.show() """
df.loc[df['model'] == 'codeqwen:7b', 'step'] -= 1

data = df.drop(columns=['individual', 'patch', 'step_duration'])
# Group by model and issue, then compute the mean, count, and standard error of the score
grouped_data = data.groupby(['model', 'issue']).agg(
    mean_score=('score', 'mean'),
    count=('score', 'size'),
    std_err=('score', lambda x: x.std() / np.sqrt(x.count()))
).reset_index()

# Pivot the data to create a matrix suitable for a heatmap
heatmap_data = grouped_data.pivot(index='issue', columns='model', values='mean_score')

# Create matrices for count and standard error for annotations
count_data = grouped_data.pivot(index='issue', columns='model', values='count')
stderr_data = grouped_data.pivot(index='issue', columns='model', values='std_err')

# Create a custom color-blind-friendly palette
cmap = sns.color_palette("viridis", as_cmap=True)

# Set the size of the plot
plt.figure(figsize=(10, 8))

# Create the heatmap using the custom color-blind-friendly palette
heatmap = sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap=cmap, 
    linewidths=.5, 
    cbar_kws={'label': 'Score'},
    xticklabels=heatmap_data.columns,
    yticklabels=heatmap_data.index
)

# Annotate the heatmap with count and standard error
for y in range(heatmap_data.shape[0]):
    for x in range(heatmap_data.shape[1]):
        plt.text(x + 0.5, y + 0.7, 
                 f'n={int(count_data.iloc[y, x])}\nSE={stderr_data.iloc[y, x]:.2f}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=6,
                 color='grey'
                 )

# Add labels and title
plt.title('LLMs on SEGym')
plt.xlabel('Model comparison')
plt.ylabel('Different issues')

# Show the plot
plt.show()

# Set the size of the plot for FacetGrid
plt.figure(figsize=(14, 10))

# Create a FacetGrid to separate the graphs for each model
g = sns.FacetGrid(data, col="model", hue="issue", col_wrap=4, height=4, aspect=1.5, palette='colorblind')

# Map the data to the FacetGrid as a lineplot
g.map(sns.lineplot, 'step', 'score')

# Add legend and adjust the layout
g.add_legend(loc='upper left')
g.set_axis_labels("Steps", "Score")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Comparison of Mean Scores by Issue Type for Each Model', fontsize=16)
g.fig.subplots_adjust(top=.9)

# Show the plot
plt.show()