import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# This script is designed to visualize the performance of LLMs
# in the context of 'Needle In A Haystack' tests.
# It displays the results of different experiments
# These CSV files can be found in the 'opencompass/outputs' directory

# Here we take the model internlm-chat-20b-hf as an example.

# Define paths to the experiment result files
results = [
    'path/to/your/first/experiment_result.csv',
    'path/to/your/second/experiment_result.csv'
]

# Process each CSV file and create a heatmap
for file_path in results:
    # Read the data
    df = pd.read_csv(file_path)

    # Process the data
    df['Context Length'] = df['dataset'].apply(lambda x: int(
        x.replace('CDME_', '').split('Depth')[0].replace('Length', '')))
    df['Document Depth'] = df['dataset'].apply(
        lambda x: float(x.replace('CDME_', '').split('Depth')[1]))
    df = df[['Document Depth', 'Context Length', 'internlm-chat-20b-hf'
             ]].rename(columns={'internlm-chat-20b-hf': 'Score'})

    # Create pivot table
    pivot_table = pd.pivot_table(df,
                                 values='Score',
                                 index=['Document Depth'],
                                 columns=['Context Length'],
                                 aggfunc='mean')

    # Create a heatmap for visualization
    cmap = LinearSegmentedColormap.from_list('custom_cmap',
                                             ['#F0496E', '#EBB839', '#0CD79F'])
    plt.figure(figsize=(17.5, 8))
    sns.heatmap(pivot_table, cmap=cmap, cbar_kws={'label': 'Score'})
    plt.title('InternLm-chat-20b 8K Context Performance\nFact Retrieval Across'
              'Context Lengths ("Needle In A Haystack")')
    plt.xlabel('Token Limit')
    plt.ylabel('Depth Percent')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the heatmap as a PNG file
    png_file_path = file_path.replace('.csv', '.png')
    plt.savefig(png_file_path)
    plt.close()  # Close the plot to prevent memory leaks
