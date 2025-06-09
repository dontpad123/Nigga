import pandas as pd

data=pd.read_csv('finds_dataset.csv')

print(data)	

# Define the Find-S algorithm function
def find_s_algorithm(data):
    """Implements the Find-S algorithm to find the most specific hypothesis."""

    # Extract feature values (all columns except the last)
    attributes = data.iloc[:, :-1].values  # e.g., Weather, Temperature, etc.

    # Extract class labels (last column)
    target = data.iloc[:, -1].values  # e.g., "Yes" or "No" (play tennis or not)

    # Step 1: Initialize the hypothesis with the first positive example ("Yes")
    for i in range(len(target)):
        if target[i] == "Yes":  # We only learn from positive examples
            hypothesis = attributes[i].copy()  # Take the first "Yes" example as initial hypothesis
            break  # Stop after finding the first one

    # Step 2: Update the hypothesis with remaining positive examples
    for i in range(len(target)):
        if target[i] == "Yes":  # Again, consider only positive examples
            for j in range(len(hypothesis)):  # Compare each attribute
                if hypothesis[j] != attributes[i][j]:  # If values differ
                    hypothesis[j] = '?'  # Generalize by replacing with '?'

    return hypothesis  # Return the final hypothesis

# Example: Run the Find-S Algorithm on a dataset called `data`
final_hypothesis = find_s_algorithm(data)

# Print the most specific hypothesis learned from positive examples
print("Most Specific Hypothesis:", final_hypothesis)
