import os

# List of the original 'lev2' files you provided
files = [
    "config_no3_autumn_lev2.py",
    "config_no3_spring_lev2.py",
    "config_no3_summer_lev2.py",
    "config_no3_winter_lev2.py",
    "config_o2_autumn_lev2.py",
    "config_o2_spring_lev2.py",
    "config_o2_summer_lev2.py",
    "config_o2_winter_lev2.py"
]

# Loop through each file in the list
for file in files:
    if "lev2" in file:
        # Read the content of the original file
        with open(file, 'r') as f:
            content = f.read()

        # Create new filenames for lev2a and lev2b
        lev2a_filename = file.replace("lev2", "lev2a")
        lev2b_filename = file.replace("lev2", "lev2b")

        # Create and write the new lev2a file
        with open(lev2a_filename, 'w') as f:
            f.write(content.replace('lev2', 'lev2a'))

        # Create and write the new lev2b file
        with open(lev2b_filename, 'w') as f:
            f.write(content.replace('lev2', 'lev2b'))

        # Remove the original file
        os.remove(file)

        # Output the new files created and original file removed
        print(f"Created {lev2a_filename} and {lev2b_filename}, and removed {file}")

