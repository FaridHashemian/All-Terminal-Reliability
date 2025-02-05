import os
import pandas as pd
import matplotlib.pyplot as plt



data_files = os.listdir('save_files')


with open('save_files/all_data.csv', 'w') as f:
    for i in range(6, 21):
        with open('save_files/results_{}.csv'.format(i), 'r') as g:
            for line in g.readlines():
                f.write('{};{}\n'.format(line.strip(), i))
        g.close()
f.close()


df = pd.read_csv('save_files/all_data.csv', delimiter=';', header=None)

if not os.path.exists('plots'):
    os.mkdir('plots')

# Get unique values of the fourth column
unique_values = df[3].unique()

# Create subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

# Flatten the axes array
axes = axes.flatten()

# Plot histograms
for ax, value in zip(axes, unique_values):
    subset = df[df[3] == value]
    ax.hist(subset[1], bins=10, alpha=0.7)
    ax.set_title(f'Histogram for node size {value}')
    ax.set_xlabel('Reliability')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)  # Set x-axis range from 0 to 1

plt.tight_layout()
#plt.show()
plt.savefig('plots/histograms.png')

plt.clf()

plt.hist(df[1], bins=10, alpha=0.7)
plt.title('Histogram for all node sizes')
plt.xlabel('Reliability')
plt.ylabel('Frequency')
plt.xlim(0, 1)
plt.savefig('plots/histogram_all.png')



average_third_column = df.groupby(3)[2].mean()
print(average_third_column)