#!/usr/bin/env python3
"""Script to convert class names to IDs in JSONL files."""

import sys
import os

# Add the current directory to the path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import convert_class_names_to_ids

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_labels.py <input_file> <output_file>")
        print("Example: python convert_labels.py input.jsonl output.jsonl")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    convert_class_names_to_ids(input_path, output_path)
