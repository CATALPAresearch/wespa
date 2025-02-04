import pandas as pd
from python.src.util import prnt
from python.src.util import print_all_output

class Extract_Sessions:
    def extract_session(self, df_textchanges):
        # Collect written text per group, pad, and until a point in time (e.g. week or session)
        reconstrcut_text = True

        df_session = df_textchanges.copy()

        df_session['moodle_pad_id'] = df_session['moodle_pad_id'].str.split('$', n=1, expand=True)

        df_session['timestamp'] = pd.to_datetime(df_session['timestamp'], unit='s')

        # Compute time since last edit for each author
        df_session['time_since_last'] = df_session.groupby('moodle_author_id')['timestamp'].diff().dt.total_seconds()

        # Identify new sessions: True if first entry or time since last > 1800s
        df_session['new_session'] = df_session['time_since_last'].isna() | (df_session['time_since_last'] > 1800)

        # Create session numbers per author
        df_session['session_nr'] = df_session.groupby('moodle_author_id')['new_session'].cumsum()

        # Assign session IDs (keeps all rows)
        df_session['session_id'] = df_session['moodle_author_id'].astype(str) + '-' + df_session['session_nr'].astype(str)

        # Filter periods
        #filtered_df = df_textchanges[df_textchanges['period'].isin(["T2", "T3"])]
        prnt(df_session.size)

        df_session.groupby('moodle_pad_id').first().sort_values(['moodle_pad_id','session_id'])[['moodle_pad_id', 'session_id']]

        return df_session

