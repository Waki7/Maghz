#!/bin/bash

pip install -e .

# Define the URLs of the files you want to download
enron="https://bailando.berkeley.edu/enron/enron_with_categories.tar.gz"
aus="https://archive.ics.uci.edu/static/public/239/legal+case+reports.zip"

# Define the directory where you want to store the downloaded and uncompressed files
download_dir="./datasets"

# Create the download directory if it doesn't exist
mkdir -p "$download_dir"

# Download and uncompress the first file
echo "Downloading and uncompressing $enron..."
wget -P "$download_dir" "$enron"
tar -xvzf "$download_dir/enron_with_categories.tar.gz" -C "$download_dir"
rm "$download_dir/enron_with_categories.tar.gz"

# Download and uncompress the second file
echo "Downloading and uncompressing $aus..."
wget -P "$download_dir" "$aus"
unzip "$download_dir/legal+case+reports.zip" -d "$download_dir"
rm "$download_dir/legal+case+reports.zip"

echo "Download and uncompression complete."