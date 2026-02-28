import tarfile

file_path = '/content/speciesnet-pytorch-v4.0.2b-v1 (1).tar.gz'

# Open the tar.gz file
with tarfile.open(file_path, 'r:gz') as tar:
    # Extract all contents to the current directory
    tar.extractall()

print(f'Successfully unzipped {file_path}')