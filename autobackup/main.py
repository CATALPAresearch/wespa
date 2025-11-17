import os
import psycopg2
import pandas as pd
import requests
import json
import sys
import subprocess
from datetime import datetime
from dotenv import load_dotenv

from extract_moodle_groups import extract_moodle_groups

load_dotenv()

# Read configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'database': os.getenv('DB_DATABASE'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

TODAY_TIMESTAMP=f"dump{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR = os.path.join('..','data',TODAY_TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)




def extract_groups_from_moodle():
    """
    Step 1. groupgrep.py: Scraped group IDs and group member IDs from a give moodle course and stores them in groups.json
    groups.json needs to be transfered to polaris://docker/backup. After updating the current course_id in the .env, the backupprocess needs to be rebild on polaris: 
    """
    # extract groups from course
    groups_data = extract_moodle_groups(
        mdl_host=os.getenv('MOODLE_HOST'), 
        mdl_course=os.getenv('MOODLE_COURSE_ID'), 
        mdl_username=os.getenv('MOODLE_USERNAME'), 
        mdl_password=os.getenv('MOODLE_PASSWORD'),
        format='data' 
        )
    
    # save file
    f = open(os.path.join(os.getcwd(), 'groups.json'), 'w')
    json.dump(groups_data, f)
    
    # send file to server
    result1 = subprocess.run(
        ['scp', './groups.json', os.getenv('SSH_PROFILE') + '://docker/backup/'],
        cwd='.', 
        check=True
        )

    if result1.returncode != 0:
        print("Error: scp failed")
        sys.exit(1)



def restart_docker_for_backup():
    """
    Step 2. Perform the backup of etherpad
    """
    sudo_password = 'hesse'
    
    command = f'cd /docker/backup && echo {os.getenv("POLARIS_PASSWORD")} | sudo -S -u {os.getenv("POLARIS_ADMIN")} docker compose up --force-recreate --build -d'
    print(command)
    return
    result = subprocess.run(
        ['ssh', os.getenv('SSH_PROFILE'), command],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0


def download_backup():
    """
    Step 3. The resulting backedup files at /docker/backup/backup/<unixtimestamp> (json file per group) need to be downloaded 
    """
    # do the backup #TODO: Reimplement in Python. Make the script executable with docker -run ...
    #sudo su burchart 
    #docker compose up --force-recreate --build
    
    # get the timestamp of the dump
    try:
        # Execute SSH command to list folders
        remote_path = '/docker/backup/backup/'
        result = subprocess.run(
            ['ssh', os.getenv('SSH_PROFILE'), f'ls -1 {remote_path}'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get folder names
        folders = result.stdout.strip().split('\n')
        
        # Filter numeric folders (timestamps) and get the latest
        timestamp_folders = [f for f in folders if f.isdigit()]
        
        if not timestamp_folders:
            print("No timestamp folders found")
            return None
        
        dump_timestamp = max(timestamp_folders, key=int)
        
        print(f"Latest backup folder on {os.getenv('SSH_PROFILE')}: {remote_path}{dump_timestamp}")
        
    except subprocess.CalledProcessError as e:
        print(f"SSH command failed: {e}")
        return None


    try: 
        # download backuped files, e.g. `scp -r polaris:/docker/backup/backup/1750059652925/ etherpad-dumps`
        result1 = subprocess.run(
            ['scp', '-r', os.getenv('SSH_PROFILE') + ':/docker/backup/backup/'+dump_timestamp+'/', '../data/etherpad-dumps'],
            cwd='.',
            check=True
            )

        if result1.returncode != 0:
            print("Error: scp failed to download the backuped etherpad data")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"SSH command failed: {e}")
        return None
    
    return dump_timestamp



def store_backup_in_postgres(dump_timestamp='1762783987869'):
    """
    Step 4. cwt_import consumes the backup files and stores them in a postgres database
    - in src/index.js den Timestamp anpassen
    - run: 
    ```
    tsc
    node --max-old-space-size=102400 ./dist/index.js "/Users/nise/Documents/proj_001_doc/pub/93-2025-CSCW-WritingAnalytics/wesepa/data/etherpad-dumps/1762783987869"
    ```
    As a result the data will be stored in cw_analytics postgres DB

    TODO: Reimplement cw_import in python. Make the script executable with docker -run ...
    """
    
    print("Step 4: Compile cwt_import using tsc")
    working_dir = "./cwt_import"
    result1 = subprocess.run(['tsc'], cwd=working_dir, check=True)

    if result1.returncode != 0:
        print("Error: tsc failed")
        sys.exit(1)

    print("\nStep 4: tsc compiled typescript \n")

    print(" Step 4: Running cwt_import...")
    result2 = subprocess.run([
        'node',
        '--max-old-space-size=102400',
        './dist/index.js',
        os.path.join(os.getenv('ETHERPAD_DUMP_FOLDER'), dump_timestamp)
    ], cwd=working_dir, check=True)

    if result2.returncode != 0:
        print("Error: node failed")
        sys.exit(1)

    print("\nStep 4: completed data import to postgres")


def export_postgres_to_csv():
    """
    Step 5. From the postgres database the tables will be extracted and stored as csv files in wespa/data/dumpYYYYMMDD
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Get all table names
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f'Exporting {table_name}...')
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            csv_path = os.path.join(OUTPUT_DIR, f'{table_name}.csv')
            df.to_csv(csv_path, index=False)

        print(f"\n All tables exported to '{OUTPUT_DIR}'")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()


def run_wespa_analysis_cycle(dump='1762783987869', semester='WS2025_26', week='week-1', scope='week'):
    """
    Step 6. Then, the csv files are processed by WESPA.
    """
    pass


def move_analysis_results_to_server():
    """
    Step 7. 
    """
    pass


def dump_couchdb_database(server_url, db_name, username, password, output_file):
    """
    Step 8. Optional step to backup couchDB
    * run `ssh -L 5984:localhost:5984 polaris`
    * open in browser: http://localhost:5984/_utils/
    * enter user and password: marc    XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3


    couchbackup --db peer_review_answer --url http://marc:XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3@localhost:5984 > peer_review_answer.json
    couchbackup --db peer_review_groupassign --url http://marc:XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3@localhost:5984 > peer_review_groupassign.json
    couchbackup --db peer_review_survey --url http://marc:XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3@localhost:5984 > peer_review_survey.json
    """
    batch_size = 99
    skip = 0
    all_docs = []
    
    print(f"Starting backup of {db_name}...")
    
    while True:
        # Get batch of documents
        url = f"{server_url}/{db_name}/_all_docs"
        params = {
            "include_docs": "true",
            "limit": batch_size,
            "skip": skip
        }
        
        try:
            response = requests.get(url, params=params, auth=(username, password))
            response.raise_for_status()
            
            data = response.json()
            rows = data.get('rows', [])
            
            if not rows:
                break
                
            # Extract documents
            batch_docs = [row['doc'] for row in rows if 'doc' in row]
            all_docs.extend(batch_docs)
            
            print(f"Backed up {len(all_docs)} documents so far...")
            
            # If we got fewer docs than requested, we're done
            if len(rows) < batch_size:
                break
                
            skip += batch_size
            
        except requests.exceptions.RequestException as e:
            print(f"Error during backup: {e}")
            sys.exit(1)
    
    # Save all documents
    backup_data = {
        "docs": all_docs,
        "total_docs": len(all_docs)
    }
    
    with open(output_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"Backup complete! {len(all_docs)} documents saved to {output_file}")


def dump_peer_review():
    """
    Step 8. Optional step to backup the peer review data
    """
    tables = [
        #"peer_review_answer",
        "peer_review_groupassign",
        "peer_review_survey"
    ]
    for table in tables:
        dump_couchdb_database(
            "http://localhost:5984",
            table,
            "marc",
            "XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3",
            f"{table}.json"
        )

def validate_completion(path1, path2):
    a = {f[6:-5] for f in os.listdir(path1) if f.startswith('group_') and f.endswith('.json')}
    b = {f[1:-5] for f in os.listdir(path2) if f.startswith('g') and f.endswith('.json')}

    print(f"Missing in B: {sorted(a - b)}\nMissing in A: {sorted(b - a)}")



if __name__ == '__main__':
    # Step 1
    #extract_groups_from_moodle()

    # Step 2
    #restart_docker_for_backup()

    # Step 3
    dump_timestamp = download_backup()

    # Step 4 (tested)
    store_backup_in_postgres(dump_timestamp=dump_timestamp)
    
    # Step 5
    export_postgres_to_csv()

    # Step 6
    current_week='week-2'
    #run_wespa_analysis_cycle(dump=dump_timestamp, semester='WS2025_26', week=current_week)

    # Step 7
    #move_analysis_results_to_server()
    #validate_completion('../data/etherpad-dumps/' + dump_timestamp, '../output/json/' + current_week)

    # ----
    # Step 8
    # dump_couchdb_database(server_url, db_name, username, password, output_file)

    # Step 9
    #dump_peer_review()

    
    
    #
    
    # couchbackup --db peer_review_answer --url http://marc:XKXTDzs4BEcU1Z6tOM6ZglNwn8sVw6mgwUnTniNTQ66D3@localhost:5984 > peer_review_answer.json
