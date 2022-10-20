"""Analyze the data of all consecutive rooms
"""

import sys
import csv

import matplotlib.pyplot as plt



def main():
    file_name = sys.argv[1]

    fitnesses = [[] for i in range(52)]
    accuracies = [[] for i in range(52)]
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            fitnesses[int(row[0])].append(float(row[1]))
            accuracies[int(row[0])].append(float(row[2]))
    
    fig, (ax_f, ax_a) = plt.subplots(figsize=(12, 8), nrows=2)
    ax_f.set_title('Correlational Score')
    ax_f.boxplot(fitnesses[1:])
    ax_a.set_title('Accuracy')
    ax_a.boxplot(accuracies[1:])
    fig.set_tight_layout(True)
    fig.savefig(file_name + 'graph.png', dpi=300)

if __name__ == '__main__':
    main()