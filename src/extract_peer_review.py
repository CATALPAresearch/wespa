#!/usr/bin/env python3
"""
Script to convert CouchDB JSON dumps to CSV files
Handles three specific file types: peer_reviewanswers, peer_review_survey, and peer_review_groupassign
"""

import json
import csv
import pandas as pd
import ast
from datetime import datetime
import argparse
import os

def load_json_file(filepath):
    """Load JSON file and return the docs array"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('docs', [])
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return []

def convert_timestamp(timestamp):
    """Convert Unix timestamp to readable datetime"""
    return timestamp
    #if timestamp:
    #    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    #return None

def parse_answer_json(answer_str):
    """Parse the JSON string in the answer field and extract key-value pairs"""
    if not answer_str:
        return {}
    
    try:
        # Parse the JSON string
        answer_dict = json.loads(answer_str)
        return answer_dict
    except (json.JSONDecodeError, TypeError):
        return {}

def parse_survey_json(survey_str):
    """Parse the survey JSON string and extract survey items"""
    if not survey_str:
        return []
    
    try:
        # Parse the JSON string
        survey_list = json.loads(survey_str)
        return survey_list
    except (json.JSONDecodeError, TypeError):
        return []

def convert_peer_reviewanswers(docs, output_file):
    """Convert peer review answers to CSV"""
    print(f"Converting {len(docs)} peer review answers...")
    
    # First pass: collect all possible answer keys
    all_answer_keys = set()
    for doc in docs:
        answer_dict = parse_answer_json(doc.get('answer', ''))
        all_answer_keys.update(answer_dict.keys())
    
    # Sort keys for consistent column order
    answer_keys = sorted(all_answer_keys)
    
    # Prepare data for CSV
    csv_data = []
    for doc in docs:
        
        row = {
            #'_id': doc.get('_id', ''),
            #'_rev': doc.get('_rev', ''),
            'projectIdooo': doc.get('projectId', ''),
            'userId': doc.get('userId', ''),
            'groupId': doc.get('groupId', ''),
            'taskId': doc.get('taskId', ''),
            'created': convert_timestamp(doc.get('created')),
            'modified': convert_timestamp(doc.get('modified')),
            #'answer_raw': doc.get('answer', '')
            'answer_raw': ''+str(doc.get('answer', '')).replace('"', '').replace(',', ';').replace('{', '').replace('}', '').rstrip()+''
        }
        
        # Add parsed answer fields
        #answer_dict = parse_answer_json(doc.get('answer', ''))
        #for key in answer_keys:
            #row[f'answer_{key}'] = answer_dict.get(key, '')
        
        csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved {len(csv_data)} records to {output_file}")
        print(f"Columns: {list(df.columns)}")

def convert_peer_review_survey(docs, output_file):
    """Convert peer review survey to CSV with wide table format for survey items"""
    print(f"Converting {len(docs)} peer review surveys...")
    
    # First pass: collect all possible survey keys from all documents
    all_survey_keys = set()
    for doc in docs:
        survey_items = parse_survey_json(doc.get('survey', ''))
        for item in survey_items:
            key = item.get('key', '')
            if key:
                all_survey_keys.add(key)
    
    # Sort keys for consistent column order
    survey_keys = sorted(all_survey_keys)
    
    csv_data = []
    survey_items_data = []
    
    for doc in docs:
        # Main survey record with survey items as separate columns
        row = {
            #'_id': doc.get('_id', ''),
            #'_rev': doc.get('_rev', ''),
            'projectId': doc.get('projectId', ''),
            'taskId': doc.get('taskId', ''),
            'created': convert_timestamp(doc.get('created')),
            'modified': convert_timestamp(doc.get('modified')),
        }
        
       #xxxx
        
        csv_data.append(row)
        
    
    # Write main survey data (wide format)
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved {len(csv_data)} survey records to {output_file}")
        print(f"Survey keys found: {survey_keys}")
    
    # Write survey items data (normalized format for reference)
    if survey_items_data:
        items_file = output_file.replace('.csv', '_items.csv')
        df_items = pd.DataFrame(survey_items_data)
        df_items.to_csv(items_file, index=False, encoding='utf-8')
        print(f"Saved {len(survey_items_data)} survey items to {items_file}")

def convert_peer_review_groupassign(docs, output_file):
    """Convert peer review group assignments to CSV"""
    print(f"Converting {len(docs)} group assignments...")
    
    csv_data = []
    for doc in docs:
        row = {
           # '_id': doc.get('_id', ''),
            #'_rev': doc.get('_rev', ''),
            'groupId': doc.get('groupId', ''),
            'taskId': doc.get('taskId', ''),
            'projectId': doc.get('projectId', ''),
            'peerId': doc.get('peerId', '')
        }
        csv_data.append(row)
    
    # Write to CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Saved {len(csv_data)} records to {output_file}")

def main():
    """Main function to process all three file types"""
    parser = argparse.ArgumentParser(description='Convert CouchDB JSON dumps to CSV files')
    parser.add_argument('--input-dir', default='.', help='Directory containing JSON files')
    parser.add_argument('--output-dir', default='.', help='Directory to save CSV files')
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # File mappings
    file_configs = [
        {
            'input': os.path.join(input_dir, 'peer_review_answer.json'),
            'output': os.path.join(output_dir, 'peer_review_answer.csv'),
            'converter': convert_peer_reviewanswers
        },
        {
            'input': os.path.join(input_dir, 'peer_review_survey.json'),
            'output': os.path.join(output_dir, 'peer_review_survey.csv'),
            'converter': convert_peer_review_survey
        },
        {
            'input': os.path.join(input_dir, 'peer_review_groupassign.json'),
            'output': os.path.join(output_dir, 'peer_review_groupassign.csv'),
            'converter': convert_peer_review_groupassign
        }
    ]
    
    # Process each file
    for config in file_configs:
        input_file = config['input']
        output_file = config['output']
        converter_func = config['converter']
        
        print(f"\n{'='*50}")
        print(f"Processing {os.path.basename(input_file)}")
        print(f"{'='*50}")
        
        # Load JSON data
        docs = load_json_file(input_file)
        
        if docs:
            # Convert to CSV
            converter_func(docs, output_file)
        else:
            print(f"No data found in {input_file}")
    
    print(f"\n{'='*50}")
    print("Conversion complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()