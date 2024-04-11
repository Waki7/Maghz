#!/bin/bash

# Check for input and output filenames
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Please provide input and output filenames. Usage: ./script.sh requirements.txt output.txt"
  exit 1
fi

input_file="$1"
output_file="$2"

# Process and write output to a new file
while IFS= read -r line; do
  # Check if the line is empty or starts with a comment
  if [[ -z "$line" || $line =~ ^\s*# ]]; then
    continue # Skip the line
  fi

  # Add quotes and a comma
  echo "'$line'," >> "$output_file"
done < "$input_file"
