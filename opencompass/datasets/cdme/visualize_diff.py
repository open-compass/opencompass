import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# This script is designed to visualize the difference in the performance of
# needleinahaystack tests with and without using guided prompts.
# It compares the results of two experiments by analyzing the differences
# between two CSV files containing the experiment outcomes.
# These CSV files can be found in the 'opencompass/outputs' directory after evaluation.

# Here we take the model internlm-chat-20b-hf as an example.

# Define paths to the experiment result files
results = ["path/to/your/first/experiment_result.csv", "path/to/your/second/experiment_result.csv"]

# Function to process a CSV file
def process_csv(file_path):
    df = pd.read_csv(file_path)
    # Extract Context Length and Document Depth from the dataset column
    df['Context Length'] = df['dataset'].apply(lambda x: int(x.split('_')[1].split('Depth')[0].replace('Length', '')))
    df['Document Depth'] = df['dataset'].apply(lambda x: float(x.split('Depth')[1]))
    return df[['Document Depth', 'Context Length', 'internlm-chat-20b-hf']].rename(columns={'internlm-chat-20b-hf': 'Score'})

# Process each CSV file
df1 = process_csv(results[0])
pivot_table1 = pd.pivot_table(df1, values='Score', index=['Document Depth'], columns=['Context Length'], aggfunc='mean')
df2 = process_csv(results[1])
pivot_table2 = pd.pivot_table(df2, values='Score', index=['Document Depth'], columns=['Context Length'], aggfunc='mean')

# Calculate the difference between the two dataframes
diff_pivot_table = pivot_table1 - pivot_table2

# Create a diverging color map from blue to red
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

# Create a heatmap to visualize the differences
plt.figure(figsize=(17.5, 8))
sns.heatmap(diff_pivot_table, cmap=cmap, center=0, cbar_kws={'label': 'Difference in Score'})
plt.title('Difference in InternLm-chat-20b 8K Context\nFact Retrieval Across Context Lengths')
plt.xlabel('Token Limit')
plt.ylabel('Depth Percent')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the heatmap to a file
plt.savefig("difference_heatmap.png")
plt.close()
