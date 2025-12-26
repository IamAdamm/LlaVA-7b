import json
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize W&B (offline mode)
wandb.init(project="llava-results", name="MultipleChoice_accuracy_analysis", mode="offline")

# Load results
results_file = '/data/llava/LLaVA/results/results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Prepare DataFrame
df = pd.DataFrame(results)

# Check correctness explicitly
def is_correct(row):
    if (row['answer'] == "A" and row['ground_truth'] == "Left") or \
       (row['answer'] == "B" and row['ground_truth'] == "Right") or \
       (row['answer'] == "C" and row['ground_truth'] == "On top") or \
       (row['answer'] == "D" and row['ground_truth'] == "Under"):
        return True
    return False

df['correct'] = df.apply(is_correct, axis=1)

# Overall accuracy
accuracy = df['correct'].mean()
print(f"Multiple Choice Accuracy: {accuracy * 100:.2f}%")
wandb.log({"overall_accuracy": accuracy})

# Accuracy per prompt type
prompt_accuracy = df.groupby('prompt_label')['correct'].mean().reset_index()

# Plot accuracy per prompt type
plt.figure(figsize=(8,5))
sns.barplot(x='prompt_label', y='correct', data=prompt_accuracy, palette="viridis")
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xlabel('Prompt Type')
plt.title('Accuracy per Prompt Type')
plt.xticks(rotation=45)
plt.tight_layout()

# Log plot to W&B
wandb.log({"accuracy_per_prompt_type": wandb.Image(plt)})
plt.show()
plt.close()

# Finish W&B logging
wandb.finish()