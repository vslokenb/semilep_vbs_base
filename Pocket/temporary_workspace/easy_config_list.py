import os

def get_formatted_filenames(directory, exclude_string):
    # List only files (not directories)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Filter out files containing the exclude string
    filtered_files = [f for f in files if exclude_string not in f]

    # Sort the files alphabetically
    filtered_files.sort()

    # Format with f-string style for file paths
    formatted_paths = [f'"{{localdir}}/datasets/{filename}",' for filename in filtered_files]

    # Remove `.json` extension for the plain names
    base_names = [os.path.splitext(f)[0] for f in filtered_files]
    base_names.sort()

    # Output
    print("Formatted file paths:")
    print('\n'.join(formatted_paths))

    print("\nBase names (comma-separated, one per line):")
    print(',\n'.join(f'"{name}"' for name in base_names))

# Example usage
directory_path = 'datasets'  # 🔁 Replace with your directory
exclude_text = 'redirector'  # 🔁 Replace with the string you want to exclude

get_formatted_filenames(directory_path, exclude_text)
