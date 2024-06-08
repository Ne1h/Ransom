import os
import xml.etree.ElementTree as ET
import pandas as pd
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Define the path to the dataset
dataset_path = r"C:\output"
output_path = r"C:\Users\Administrator\Documents\2extracted_features.csv"

# Initialize an empty list to hold the data
data = []

# Initialize sets to hold all unique permissions, intents, and keywords
all_permissions = set()
all_intents = set()
all_keywords_text = set()
all_keywords_images = set()

# Function to perform NLP processing
def process_text(text):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    processed_text = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
        processed_text.extend(processed_tokens)
    return processed_text

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

                # Extract text from res/values and res/layout
                keywords_text = []
                for xml_file in ['res/values/strings.xml', 'res/layout/activity_main.xml']:
                    xml_path = os.path.join(root, xml_file)
                    if os.path.exists(xml_path):
                        xml_tree = ET.parse(xml_path)
                        for elem in xml_tree.iter():
                            if elem.text:
                                keywords_text.extend(process_text(elem.text))

                # Extract text from images in res/drawable using OCR
                keywords_images = []
                drawable_path = os.path.join(root, 'res/drawable')
                if os.path.exists(drawable_path):
                    for image_file in os.listdir(drawable_path):
                        if image_file.endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(drawable_path, image_file)
                            text = pytesseract.image_to_string(Image.open(image_path))
                            keywords_images.extend(process_text(text))

                # Update the sets of all unique permissions, intents, and keywords
                all_permissions.update(permissions)
                all_intents.update(intents)
                all_keywords_text.update(keywords_text)
                all_keywords_images.update(keywords_images)

                # Append the file name, permissions, intents, and keywords to data
                data.append((apk_file_name, permissions, intents, keywords_text, keywords_images))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Prepare data for DataFrame creation
rows = []
for file_name, permissions, intents, keywords_text, keywords_images in data:
    row = {'File': file_name}
    for permission in permissions:
        row[permission] = 1
    for intent in intents:
        row[intent] = 1
    rows.append(row)

# Create a DataFrame with columns: 'File' + sorted list of all unique permissions + sorted list of all unique intents
columns = ['File'] + sorted(all_permissions) + sorted(all_intents)
df = pd.DataFrame(rows, columns=columns)

# Create a TfidfVectorizer for text and image keywords
tfidf_vectorizer_text = TfidfVectorizer(max_features=500, ngram_range=(2, 3))
tfidf_vectorizer_image = TfidfVectorizer(max_features=100, ngram_range=(2, 3))

# Concatenate all text and image keywords for TF-IDF fitting
all_texts = [' '.join(keywords_text) for _, _, _, keywords_text, _ in data]
all_images = [' '.join(keywords_images) for _, _, _, _, keywords_images in data]
tfidf_vectorizer_text.fit(all_texts)
tfidf_vectorizer_image.fit(all_images)

# Transform text and image keywords to TF-IDF features
text_features = tfidf_vectorizer_text.transform(all_texts).toarray()
image_features = tfidf_vectorizer_image.transform(all_images).toarray()

# Binarize the features by setting a threshold
text_features_binary = (text_features > 0).astype(int)
image_features_binary = (image_features > 0).astype(int)

# Add binary features for text and images to rows
rows = []
for i, (file_name, permissions, intents, _, _) in enumerate(data):
    row = {'File': file_name}
    for permission in permissions:
        row[permission] = 1
    for intent in intents:
        row[intent] = 1
    
    # Add binary features for text
    for j, feature in enumerate(tfidf_vectorizer_text.get_feature_names_out()):
        row[f"text_{feature}"] = text_features_binary[i, j]
    
    # Add binary features for images
    for j, feature in enumerate(tfidf_vectorizer_image.get_feature_names_out()):
        row[f"image_{feature}"] = image_features_binary[i, j]
    
    rows.append(row)

# Create the final DataFrame with permissions, intents, and features
df = pd.DataFrame(rows, columns=columns + [f"text_{feature}" for feature in tfidf_vectorizer_text.get_feature_names_out()] + [f"image_{feature}" for feature in tfidf_vectorizer_image.get_feature_names_out()])

# Fill NaN values with 0 (indicating absence of permission or intent)
df = df.fillna(0)

# Save the final DataFrame to a CSV file
df.to_csv(output_path, index=False)

# Print the number of unique permissions and intents
print(f"Number of unique permissions: {len(all_permissions)}")
print(f"Number of unique intents: {len(all_intents)}")
print(f"Number of selected features from text: {len(tfidf_vectorizer_text.get_feature_names_out())}")
print(f"Number of selected features from images: {len(tfidf_vectorizer_image.get_feature_names_out())}")

print(f"Permissions, intents, and features extracted and saved to {output_path}")
