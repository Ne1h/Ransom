import os
import xml.etree.ElementTree as ET
import pandas as pd
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

# Define the path to the dataset
dataset_path = r"C:\output"

# Initialize an empty list to hold the data
data = []

# Initialize sets to hold all unique permissions and intents
all_permissions = set()
all_intents = set()

# Walk through all subfolders
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file == "AndroidManifest.xml":
            file_path = os.path.join(root, file)
            apk_file_name = os.path.basename(root) + ".apk"  # Assuming the folder name is the APK file name
            try:
                tree = ET.parse(file_path)
                manifest_root = tree.getroot()
                
                # Extract permissions
                permissions = set()
                for elem in manifest_root.findall('uses-permission'):
                    permission = elem.get('{http://schemas.android.com/apk/res/android}name')
                    if permission:
                        permissions.add(permission)
                
                # Extract intents
                intents = set()
                for elem in manifest_root.findall('.//intent-filter//action'):
                    intent = elem.get('{http://schemas.android.com/apk/res/android}name')
                    if intent:
                        intents.add(intent)
                
                # Update the sets of all unique permissions and intents
                all_permissions.update(permissions)
                all_intents.update(intents)
                
                # Append the file name, permissions, and intents to data
                data.append((apk_file_name, permissions, intents))
            except ET.ParseError:
                print(f"Error parsing {file_path}")

# Convert the data to a list of dictionaries
rows = []
for file_name, permissions, intents in data:
    row = {'File': file_name}
    for permission in permissions:
        row[permission] = 1
    for intent in intents:
        row[intent] = 1
    rows.append(row)

# Create a DataFrame with columns: 'File' + sorted list of all unique permissions + sorted list of all unique intents
columns = ['File'] + sorted(all_permissions) + sorted(all_intents)
df = pd.DataFrame(rows, columns=columns)

# Fill NaN values with 0 (indicating absence of permission or intent)
df = df.fillna(0)

# Save the DataFrame to a CSV file
output_path = r"C:\Users\Administrator\Documents\permissions_and_intents.csv"
df.to_csv(output_path, index=False)

# Print the number of unique permissions and intents
print(f"Number of unique permissions: {len(all_permissions)}")
print(f"Number of unique intents: {len(all_intents)}")

print(f"Permissions and intents extracted and saved to {output_path}")
