# PTM-sumo


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, matthews_corrcoef, confusion_matrix
import joblib

# Load positive and negative samples
positive_samples = pd.read_excel('SVMpos.xlsx', header=None)
negative_samples = pd.read_excel('features-negnonlyssites-nona.xlsx', header=None)

# Create a target column (1 for positive samples, 0 for negative samples)
positive_samples['label'] = 1
negative_samples['label'] = 0

# Concatenate positive and negative samples into a single DataFrame
data = pd.concat([positive_samples, negative_samples], axis=0)

# Split the data into features (X) and labels (y)
X = data.iloc[:, :-1]  # Features
y = data['label']      # Labels

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(probability=True)  # Enable probability estimates for ROC curve later

# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

# Calculate Matthews correlation coefficient (MCC)
mcc = matthews_corrcoef(y_test, y_pred)
print(f"MCC: {mcc:.2f}")

# Visualize the decision boundary (works for 2D feature spaces)
if X_train.shape[1] == 2:
    # Create a meshgrid of points to visualize decision boundary
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Visualize decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=plt.cm.RdBu, marker='o')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary")
    plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['Actual 0', 'Actual 1'])

# Annotate the confusion matrix with the counts
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')

plt.tight_layout()
plt.show()

# Save the trained model to a file
joblib.dump(svm_classifier, 'svm_model.pkl')




##finding atoms distance

def find_atoms_within_distance(pdb_id, chain_id, residue_number, distance_cutoff):
    # Step 1: Retrieve the PDB file
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(pdb_id, f"{pdb_id}.pdb")

    # Step 2: Extract all heavy atoms
    heavy_atoms = Selection.unfold_entities(structure, "A")

    # Step 3: Calculate distances and filter by cutoff
    target_residue = structure[0][chain_id][residue_number]
    nearby_atoms = []
    for atom in heavy_atoms:
        distance = atom - target_residue["CA"]  # Calculate distance to C-alpha atom
        if distance <= distance_cutoff:
            nearby_atoms.append(atom)

    return nearby_atoms

# Usage example
pdb_id = "2o61_37-51"
chain_id = "B"
residue_number = 37
distance_cutoff = 5.0

nearby_atoms = find_atoms_within_distance(pdb_id, chain_id, residue_number, distance_cutoff)
for atom in nearby_atoms:
    print(atom.get_full_id())



##lowest resolution
from Bio.PDB import PDBList
import os
import pandas as pd

def read_pdb_res(pdb_id):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir='.')
    resolution = None
    pdb_file_path = f'pdb{pdb_id.lower()}.ent'
    new_pdb_file_path = f'{pdb_id.lower()}.pdb'
    os.rename(pdb_file_path, new_pdb_file_path)

    with open(new_pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith('REMARK   2 RESOLUTION'):
                resolution_value = line[23:30].strip()
                if resolution_value == 'N/A':
                    resolution = float('inf')  # Use a high value for 'N/A'
                else:
                    try:
                        resolution = float(resolution_value)
                    except ValueError:
                        resolution = float('inf')
    return resolution

data = {
    'UniprotID': ['A0AVK6', 'A0AVK6', 'A4UGR9', 'A5YKK6', 'A5YKK6', 'A5YKK6', 'A5YKK6', 'A5YKK6', 'A5YKK6'],
    'PDBID': ['4YO2', '4YO2', '4F14', '4C0D', '4CQO', '4CRU', '4CRV', '4CRW', '4CT4'],
    'Position': [858, 470, 137, 1163, 1163, 1163, 1163, 1163, 1163],
    'ExperimentType': ['x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction', 'x-ray diffraction']
}

df = pd.DataFrame(data)
df['Resolution'] = df['PDBID'].apply(read_pdb_res)
print(df)



##Range distance

from Bio.PDB import PDBParser, Selection
import pandas as pd

def find_atoms_within_distance(pdb_id, chain_id, residue_number, distance_ranges):
    # Step 1: Retrieve the PDB file
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure(pdb_id, f"{pdb_id}.pdb")

    # Step 2: Extract all heavy atoms
    heavy_atoms = Selection.unfold_entities(structure, "A")

    # Step 3: Get the target residue's C-alpha atom
    target_residue = structure[0][chain_id][residue_number]["CA"]

    # Step 4: Find nearby atoms for each distance range
    nearby_atoms_by_range = {f"{min_d} to {max_d}": [] for min_d, max_d in distance_ranges}
    for atom in heavy_atoms:
        distance = atom - target_residue  # Calculate distance to C-alpha atom
        for min_d, max_d in distance_ranges:
            if min_d < distance < max_d:  # Exclude distance 0.0
                nearby_atoms_by_range[f"{min_d} to {max_d}"].append(atom)

    return nearby_atoms_by_range

def main():
    # Define distance ranges excluding distance 0.0
    distance_ranges = [
        (0.5, 1.0),
        (1.0, 1.5),
        (1.5, 2.0),
        (2.0, 2.5),
        (2.5, 3.0),
        (3.0, 3.5),
        (3.5, 4.0),
        (4.0, 4.5),
        (4.5, 5.0)
    ]

    # Define multiple PDB IDs, chain IDs, and residue numbers
    pdb_data = [
        {"pdb_id": "2o61", "chain_id": "B", "residue_number": 37},
        {"pdb_id": "1wm2", "chain_id": "A", "residue_number": 61},
        {"pdb_id": "2o61_37-51", "chain_id": "B", "residue_number": 37}


        # Add more PDB data as needed
    ]

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["PDB ID", "Chain ID", "Residue Number"] + [f"{min_d} to {max_d}" for min_d, max_d in distance_ranges])

    for data in pdb_data:
        pdb_id = data["pdb_id"]
        chain_id = data["chain_id"]
        residue_number = data["residue_number"]

        nearby_atoms_by_range = find_atoms_within_distance(pdb_id, chain_id, residue_number, distance_ranges)

        # Append the results to the DataFrame
        result_row = {"PDB ID": pdb_id, "Chain ID": chain_id, "Residue Number": residue_number}
        for distance_range, nearby_atoms in nearby_atoms_by_range.items():
            result_row[f"{distance_range}"] = len(nearby_atoms)
        results_df = results_df.append(result_row, ignore_index=True)

    print(results_df)
    results_df.to_csv('results.csv')

if __name__ == "__main__":
    main()
