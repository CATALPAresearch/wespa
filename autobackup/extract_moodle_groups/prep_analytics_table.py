"""
This script creates the postgres table structur as an sql file. The structure is derived from the csv files of existing imports. 
The output as sql file can be used to create the table structure for an empty postgres database, e.g. cw_analytics
"""

import os
import csv

# Folder containing the CSV files
CSV_FOLDER = '/Users/nise/Documents/proj_001_doc/pub/93-2024-CSCW-WritingAnalytics/wesepa/data/dump20240826'  # adjust to your folder path
OUTPUT_FILE = 'create_tables.sql'

def sanitize_identifier(name):
    """Sanitize table/column names for SQL"""
    return name.strip().replace(' ', '_').replace('-', '_').lower()

def csv_to_create_table_statement(csv_path):
    table_name = sanitize_identifier(os.path.splitext(os.path.basename(csv_path))[0])
    
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)

    if not header:
        return None  # skip empty files

    columns = [f'"{sanitize_identifier(col)}" TEXT' for col in header]
    column_defs = ',\n  '.join(columns)

    return f'CREATE TABLE "{table_name}" (\n  {column_defs}\n);'

def main():
    statements = []

    for file in os.listdir(CSV_FOLDER):
        if file.endswith('.csv'):
            csv_path = os.path.join(CSV_FOLDER, file)
            stmt = csv_to_create_table_statement(csv_path)
            if stmt:
                statements.append(stmt)

    # Write all statements to a SQL file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write('\n\n'.join(statements))

    print(f"✅ SQL statements written to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
