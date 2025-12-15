"""
This script replaces a string in all json files that are located in one give folder
"""

import os
import json
from pathlib import Path

def replace_in_json_files(folder_path):
    """
    Replace "moodle_group_id" with "group" in all JSON files in the specified folder.
    
    Args:
        folder_path: Path to the folder containing JSON files
    """
    folder = Path(folder_path)
    
    # Find all JSON files in the folder
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    print(f"Found {len(json_files)} JSON file(s)")
    
    for json_file in json_files:
        try:
            # Read the file content as text
            with open(json_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the string
            modified_content = content.replace('"moodle_group_id"', '"group"')
            
            # Write back to the file
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✓ Processed: {json_file.name}")
            
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")

if __name__ == "__main__":
    # Specify your folder path here
    folder_path = "/Users/nise/Documents/proj_001_doc/pub/93-2025-CSCW-WritingAnalytics/wesepa/output/json/week-3/tmp"  # Current directory, change this to your folder path
    
    replace_in_json_files(folder_path)
    print("\nDone!")