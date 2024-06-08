import os
import subprocess
import zipfile

# Directory containing the APK files
apk_directory = r"C:\Benign"  # APK files directory
# Directory to store the extracted files
output_directory = r"C:\output_benign"  # output directory
# Path to the CFR jar file
cfr_jar_path = r"C:\Repo\Ransomdroid\cfr-0.152.jar"  # Update this to the path where you have cfr.jar

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

def extract_dex_files(apk_path, output_path):
    dex_files = []
    with zipfile.ZipFile(apk_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.dex'):
                dex_file_path = zip_ref.extract(file, output_path)
                dex_files.append(dex_file_path)
                print(f"Extracted {file} to {output_path}")
    return dex_files

def convert_dex_to_jar(dex_file, output_path):
    jar_output_path = os.path.join(output_path, os.path.splitext(os.path.basename(dex_file))[0] + '.jar')
    dex2jar_command = f"d2j-dex2jar \"{dex_file}\" -o \"{jar_output_path}\""
    print(f"Running: {dex2jar_command}")  # Print the command being executed
    subprocess.run(dex2jar_command, shell=True, stdin=subprocess.PIPE)  # Execute the command
    print(f"Converted {dex_file} to {jar_output_path}")
    return jar_output_path

def decompile_jar_to_java(jar_file, output_path):
    java_output_path = os.path.join(output_path, "java_sources")
    os.makedirs(java_output_path, exist_ok=True)
    cfr_command = f"java -jar \"{cfr_jar_path}\" \"{jar_file}\" --outputdir \"{java_output_path}\""
    print(f"Running: {cfr_command}")  # Print the command being executed
    subprocess.run(cfr_command, shell=True, stdin=subprocess.PIPE)  # Execute the command
    print(f"Decompiled {jar_file} to {java_output_path}")

# Iterate through all subdirectories and files in the specified directory
for root, dirs, files in os.walk(apk_directory):
    for file in files:
        if file.endswith(".apk"):
            apk_path = os.path.join(root, file)  # Full path to the APK file
            relative_path = os.path.relpath(root, apk_directory)  # Relative path for output directory structure
            output_path = os.path.join(output_directory, relative_path, os.path.splitext(file)[0])  # Output path

            # Check if the output directory for this APK file already exists
            if os.path.exists(output_path) and os.listdir(output_path):
                print(f"Skipping {apk_path}, already processed.")
                continue

            # Ensure the output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Debugging print statements
            print(f"Processing file: {apk_path}")
            print(f"Output path: {output_path}")

            # Run the apktool command to extract AndroidManifest.xml and res folder
            apktool_command = f"apktool d -f -q \"{apk_path}\" -o \"{output_path}\""
            print(f"Running: {apktool_command}")  # Print the command being executed
            subprocess.run(apktool_command, shell=True, stdin=subprocess.PIPE)  # Execute the command

            # Extract .dex files
            dex_files = extract_dex_files(apk_path, output_path)

            # Convert .dex files to .jar
            for dex_file in dex_files:
                jar_file = convert_dex_to_jar(dex_file, output_path)
                
                # Decompile .jar files to Java source code
                decompile_jar_to_java(jar_file, output_path)

print("Extraction, conversion, and decompilation completed.")
