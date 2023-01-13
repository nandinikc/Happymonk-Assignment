import numpy as np

# Example datasets A and B
A = [1, 2, 3, 4, 5, 6]
B = [2, 3, 4, 5, 6, 7]

# Calculate Pearson correlation coefficient
corr = np.corrcoef(A, B)[0, 1]

# Print the correlation coefficient
print(corr)