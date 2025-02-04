"""
 curl -X POST "http://127.0.0.1:5000/get_jobs/" \                   
     -H "Authorization: Bearer test" \
     -H "Content-Type: application/json" \
     -d '{"group_id": 1, "user_id": 42, "action": "login"}'
"""
from python.src.flaskk import Flask, request, jsonify
#from flask_restx import Api, Resource, fields
#from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from rq import Queue, Worker
import threading
import json
import sqlite3
import datetime

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize RQ for background tasks
r = None  # To be set in a worker thread
queue_name = 'default'
worker_thread = None

def get_worker():
    global r
    if not hasattr(r, 'get_queue'):
        from rq import Queue, Worker
        
        def __init__(self):
            self.get_queue = False
            
        # Initialize
        q = Queue(queue_name)
        w = Worker(q)
        
        r = (w, q)
        r.get_queue = True
        
    w, q = r
    
def run():    
    while not w.stop:
        job = next(w)

def create_jobs_table():
    conn = sqlite3.connect('jobs.db')

    # Prepare a cursor object
    cursor = conn.cursor()
    try:
        # Create table if not exists
        cursor.execute("CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY AUTOINCREMENT,job_name TEXT NOT NULL, description TEXT,parameters JSONB default '{}',status TEXT DEFAULT 'idle',start_time DATETIME DEFAULT CURRENT_TIMESTAMP,end_time DATETIME,summary TEXT)")
    except sqlite3.OperationalError as e:
        print("Error creating table", e)

    # Commit changes and close connection
    conn.commit()
    conn.close()


@app.route('/submit_job', methods=['POST'])
def submit_job():
    # Extract form data if needed or use request.json
    print("Submitted job:", request.json)
    
    # Create new job record in database
    conn = sqlite3.connect('jobs.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO jobs'
        '(job_name, description, parameters, status, start_time)'
        ' VALUES(?, ?, ?, ?, ?)',
        (
            request.json['name'], 
            request.json['description'],
            json.dumps(request.json.get('parameters', {})),
         'idle',
         datetime.datetime.now())
    )
    conn.commit()
    conn.close()

    # start the job
    #TODO

    # Send response
    return jsonify({'status': 'success'}), 201


@app.route('/get_jobs')
def get_jobs():
    cursor = sqlite3.connect('jobs.db').cursor()
    cursor.execute("SELECT * FROM jobs")
    rows = cursor.fetchall()
    
    if not rows:
        return jsonify({'jobs': []}), 200
    
    # Convert tuples to dictionaries
    jobs = []
    for row in rows:
        job_dict = {
            'id': row[0],
            'job_name': row[1],
            'description': row[2],
            'parameters': json.loads(row[3]),
            'status': row[4],
            'start_time': row[5],
            'end_time': row[6],
            'summary': row[7]
        }
        jobs.append(job_dict)
    
    return jsonify({'jobs': jobs}), 200


@app.route('/')
def home():
    return "Hello, Flask + WebSockets"

@socketio.on('connection')
def test_connection():
    print("New client connected")

@socketio.on('close')
def close_connection(*args):
    print("Client disconnected")



def initialize_rq():
    queue_name = 'default'
    worker_thread = threading.Thread(target=run_worker, args=(queue_name,))
    worker_thread.start()


if __name__ == '__main__':
    create_jobs_table()
    app.run(debug=True) 
    socketio.run(app)
    # Initialize RQ workers before submitting jobs
    initialize_rq()