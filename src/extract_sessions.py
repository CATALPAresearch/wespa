import pandas as pd
import numpy as np
from src.util import prnt
from src.util import print_all_output

class Extract_Sessions:

    def __init__(self, semester):
        self.semester = semester
        pass

    def load_and_combine_df(self):
        """..."""
        output_path = './output'
        project_name = 'edm25'
        # load
        df_comments = pd.read_csv(f'{output_path}/{project_name}-{semester}-01-data-comments.csv')
        df_comment_replies = pd.read_csv(f'{output_path}/{project_name}-{semester}-01-data-comment-replies.csv')
        df_chats = pd.read_csv(f'{output_path}/{project_name}-{semester}-01-data-chats.csv')
        df_scrolls = pd.read_csv(f'{output_path}/{project_name}-{semester}-01-data-scrolls.csv')
        df_textedits = pd.read_csv(f'{output_path}/{project_name}-{semester}-01-data-textedits.csv')

        

        # select and combine
        cols = ['id','authorid', 'groupid', 'padid', 'timestamp', 'type']
        df_sel_comments = df_comments[cols]
        df_sel_comment_replies = df_comment_replies[cols]
        df_sel_chats = df_chats[cols]
        df_sel_scrolls = df_scrolls[cols]
        df_sel_textedits = df_textedits[['id','moodle_author_id','moodle_group_id','moodle_pad_id', 'timestamp', 'type']]
        df_sel_textedits = df_sel_textedits.rename(columns={
            'id': 'id',
            'moodle_author_id': 'authorid',
            'moodle_group_id': 'groupid',
            'moodle_pad_id': 'padid',
            'timestamp': 'timestamp',
            'type': 'type',
        })
        #df = pd.DataFrame()
        df = pd.concat([df_sel_comments, df_sel_comment_replies, df_sel_chats, df_sel_scrolls, df_sel_textedits])
        return df
    
    def extract_sessions(self, df_textchanges):
        """..."""
        df_session = df_textchanges.copy()
        #TO observe
        df_session['moodle_author_id'] = df_session['moodle_author_id'].replace([np.inf, -np.inf], np.nan, inplace=True) 
        df_session['moodle_author_id'] = df_session['moodle_author_id'].astype(str)
        df_session['moodle_author_id'] = df_session['moodle_author_id'].astype(int)
        pad_split = df_session['moodle_pad_id'].str.split('$', n=1, expand=True)
        df_session['moodle_pad_id'] = pad_split[0] if len(pad_split) > 0 else df_session['moodle_pad_id']

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
        #prnt(df_session.size)

        #df_session.groupby('moodle_pad_id').first().sort_values(['moodle_pad_id','session_id'])[['moodle_pad_id', 'session_id']]

        # rearrange columns
        cols_to_order = ['session_id', 'id','moodle_author_id']
        new_columns = cols_to_order + (df_session.columns.drop(cols_to_order).tolist())
        df_session = df_session[new_columns]
        return df_session.sort_values(by=['session_id'], ascending=False)
    

    def save_to_file(self):
        """..."""
        pass