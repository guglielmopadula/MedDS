import os
import glob

# Specify the directory to search
directory = '.'  # Update with your directory

# Use glob to find all Python files starting with 'config' in the specified directory and subdirectories
files = glob.glob(os.path.join(directory, '**', 'config*.py'), recursive=True)

# Loop through each file and remove the last double quote if it exists
for file in files:
    try:
        # Read the content of the file
        with open(file, 'r') as f:
            content = f.read()

        # Check if the content ends with a double quote and remove it
        if content.endswith('"'):
            content = content[:-1]

            # Write the updated content back to the file
            with open(file, 'w') as f:
                f.write(content)

            print(f"Last double quote removed from {file}")
        else:
            print(f"No double quote")
    
    except Exception as e:
        print(f"Error processing {file}: {e}")