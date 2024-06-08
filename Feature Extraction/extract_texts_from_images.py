import pandas as pd
import os
import xml.etree.ElementTree as ET
from PIL import Image
import pytesseract

# Function to extract text from XML files
def extract_texts_from_xml(root_folder):
    data = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file in ["activity_main.xml", "strings.xml"]:
                file_path = os.path.join(root, file)
                apk_file_name = os.path.basename(root) + ".apk"  # Assuming the folder name is the APK file name
                try:
                    tree = ET.parse(file_path)
                    xml_root = tree.getroot()
                    
                    # Extract texts
                    texts = []
                    for elem in xml_root.iter():
                        text = elem.text
                        if text and text.strip():
                            texts.append(text.strip())
                    
                    # Append the file name and texts to data
                    data.append((apk_file_name, file, " ".join(texts)))
                except ET.ParseError:
                    print(f"Error parsing {file_path}")
    return data

# Function to extract text from images using OCR
def extract_texts_from_images(images_folder):
    image_data = []
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    text = pytesseract.image_to_string(Image.open(file_path), lang='eng+rus+chi_sim')
                    image_data.append((file, text))
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return image_data

# Paths to the dataset and images
dataset_path = r"C:\Users\Administrator\Downloads\Dataset_\Datasets2\Adware2"
images_path = r"C:\Users\Administrator\Downloads\Dataset_\Images"

# Extract texts from XML and images
xml_texts = extract_texts_from_xml(dataset_path)
image_texts = extract_texts_from_images(images_path)

# Combine data into a DataFrame
df_xml = pd.DataFrame(xml_texts, columns=['File', 'XML_File', 'Texts'])
df_images = pd.DataFrame(image_texts, columns=['Image_File', 'Extracted_Text'])

# Save DataFrames to CSV files
df_xml.to_csv(r"C:\Users\Administrator\Documents\extracted_texts_from_xml.csv", index=False)
df_images.to_csv(r"C:\Users\Administrator\Documents\extracted_texts_from_images.csv", index=False)

print("Texts extracted from XML and images have been saved to CSV files.")
