import os
import pandas as pd
import numpy as np

from .settings import *
from .util import prnt

class Load:

    def __init__(self, data_path='../data/'):
        # inti vars form the settings file
        self.data_path = data_path
        self.output_path = outputpath
        self.dump = dump
        self.semester = semester
        self.teacher_ids = teacher_ids  
        self.period_1 = period_1
        self.period_2 = period_2
        self.period_3 = period_3

        self.df = None
    
    def load_data(self, filename):
        return pd.read_csv(f'{self.data_path}/{self.dump}/{filename}')
    

    def preprocess(self, df, activity, filter_group=None, filter_weeks=None):
        """ Convert column format and filter data """
        print(df.columns)
        if filter_group != None:# and type(filter_group).__name__ in ('list', 'tuple'):
            print('filter-'+activity)
            df = df[df['groupid'].isin(filter_group)]
        try:
            df['timecreated'] = pd.to_datetime(
                df.timestamp, 
                errors='raise',
                utc=True, 
                format='ISO8601' #format="YYYY-MM-DD HH:MM:SS.ss±TZ"
                ) 
            df['timestamp'] = df.timecreated.apply(lambda x: pd.Timestamp.timestamp(x))
            df['week'] = df['timecreated'].apply(lambda x: x.strftime("%y-%U")) 
        except():
            print('EXEPTION at time conversion')

        # FixMe: The periods 1-3 are too specific
        df.assign(period='none')
        if filter_weeks != None:    
            df['period'] = df['week'].apply(lambda x: 'T1' if x in self.period_1 else ('T2' if x in self.period_2 else ('T3' if x in self.period_3 else 'other')))
            df = df[df['period'].isin(['T2', 'T3'])]
            
        #df.loc[:, 'activity'] = activity
        #df.loc.__setitem__((slice(None), ('activity')), activity)
        #df.loc[:,'activity'] = activity
        #df['activity'].apply(lambda x: activity)
        df.assign(activity=activity)
        #df.__getitem__('activity').__setitem__('activity', activity)
        if 'text' in df.columns:
            df.assign(text_length=0)
            df.loc[:, 'text_length'] = df.loc[:, 'text'].str.len() # Fixme
        
        # remove teachers 
        df = df[~df['authorid'].isin(self.teacher_ids)]
        df = df[df['authorid'] != '']

        # rearrange columns
        cols_to_order = ['id','authorid', 'groupid', 'padid']
        new_columns = cols_to_order + (df.columns.drop(cols_to_order).tolist())
        df = df[new_columns]
        
        return df
        

    def summarize(self, df, sum_col_name):
        """ summarize data """
        return df.groupby(['authorid', 'groupid', 'period']).agg(
            sum_entries=('authorid', 'count'),
            sum_length=(sum_col_name, 'sum')
        ).reset_index()
    

    def process_comments(self, filter_group=None, filter_weeks=None):
        print(f'Loading comment data from CSV')
        df = self.preprocess(self.load_data('pad_comment.csv'), 'comment', filter_group, filter_weeks)
        print(df.shape)
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-comment.csv')
        self.save_data(df_summary, 'data-summary-comment.csv')
    
    def process_comment_replies(self, filter_group=None, filter_weeks=None):
        print(f'Loading comment replies data from CSV')
        df = self.preprocess(self.load_data('pad_comment_reply.csv'), 'commentreply', filter_group, filter_weeks)
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-reply.csv')
        self.save_data(df_summary, 'data-summary-reply.csv')
    
    def process_chat(self, filter_group=None, filter_weeks=None):
        print(f'Loading chat data from CSV')
        df = self.preprocess(self.load_data('pad_chat.csv'), 'chat', filter_group, filter_weeks)
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-chat.csv')
        self.save_data(df_summary, 'data-summary-chat.csv')
    

    def process_scrolls(self, filter_group= None, filter_weeks=None):
        print(f'Loading scroll data from CSV')
        df = self.preprocess(self.load_data('pad_scrolling.csv'), 'scroll', filter_group, filter_weeks)
        df[['top', 'bottom']] = df[['top', 'bottom']].fillna(0)
        df['top_prev'] = df['top'].shift(fill_value=0)
        df['distance_tmp'] = df['top'] - df['top_prev']
        df['distance'] = df['distance_tmp'].abs()
        df['distance_down'] = df['distance_tmp'].clip(lower=0)
        df['distance_up'] = df['distance_tmp'].clip(upper=0)
        df = df.drop('distance_tmp', axis=1)
        
        df_summary = df.groupby(['authorid', 'period']).agg(
            sum_scroll_events=('authorid', 'count'),
            sum_scroll_distance=('distance', 'sum'),
            sum_scroll_distance_up=('distance_up', 'sum'),
            sum_scroll_distance_down=('distance_down', 'sum')
        ).reset_index()
        
        self.save_data(df, 'data-scroll.csv')
        self.save_data(df_summary, 'data-summary-scroll.csv')
    

    def process_textedits(self, filter_group=None, filter_weeks=None):
        print(f'Loading textedit data from CSV')
        df = self.preprocess(self.load_data('pad_commit.csv'), 'textedit', filter_group, filter_weeks)
        
        # Step 2: Rename columns for consistency
        df = df.rename(columns={
            'userid': 'moodle_author_id',
            'groupid': 'moodle_group_id',
            'padid': 'moodle_pad_id',
            'taskid': 'moodle_task_id',
            'text': 'textedit_changeset',
            'rev': 'textedit_rev',
            'type': ''
        })
        df["type"] = "textedit"
        
        # Step 6: Hash Moodle user IDs (if enabled)
        if hashit:
            df = self.hashUsers(df)
        
        # Step 8: Remove groups with only one active member
        active_groups = df.groupby("moodle_group_id")['moodle_author_id'].nunique()
        df = df[df['moodle_group_id'].isin(active_groups[active_groups > 1].index)]
        
        # Step 9: Select only required columns
        selected_columns = [
            'moodle_author_id', 'moodle_group_id', 'moodle_pad_id',
            'textedit_changeset', 'timestamp', 'week', 'period', 'type'
        ]
        df = df[selected_columns]

        self.save_data(df, 'data-textedit.csv')
        self.df = df


    def get_date(self):
        return self.df    

    def save_data(self, df, filename):
        """ Save data to CSV file"""
        df.to_csv(
            f'{self.output_path}/{project_name}-{self.semester}-01-{filename}', 
            index=False,
            quotechar='"'
            )
    

    def run(self, filter_group=None, filter_weeks=None):
        """ Run all preprocessing tasks """
        self.process_textedits(filter_group, filter_weeks)
        self.process_comments(filter_group, filter_weeks)
        self.process_comment_replies(filter_group, filter_weeks)
        self.process_chat(filter_group, filter_weeks)
        self.process_scrolls(filter_group, filter_weeks)
    

    def hashUsers(self, df):
        """ TODO hash """
        def load_mapping(file_path, cols):
            if os.path.exists(file_path):
                return pd.read_csv(file_path, delimiter=';', skiprows=1, names=cols, encoding='utf-8')
            else:
                print(f"Error: File {file_path} does not exist.")
                return pd.DataFrame()

        moodle_etherpad_file = f'../data/{semester}_user_mapping_moodle_etherpad.csv'
        etherpad_hash_file = f'../data/{semester}_user_mapping_etherpad_hash.csv'
        moodle_hash_file = f'../data/{semester}_user_mapping_moodle_hash.csv'

        u_moodle_etherpad = load_mapping(moodle_etherpad_file, ["moodle_user_id", "etherpad_user_id"])
        if not u_moodle_etherpad.empty:
            df = df.merge(u_moodle_etherpad, left_on="moodle_author_id", right_on="moodle_user_id", how="right")
        
        u_etherpad_hash = load_mapping(etherpad_hash_file, ["etherpad_user_id", "hashuser", "hashgroup", "groupcat"])
        if not u_etherpad_hash.empty:
            df = df.merge(u_etherpad_hash, on="etherpad_user_id", how="right")
        
        u_moodle_hash = load_mapping(moodle_hash_file, ["moodle_user_id", "hashuser", "hashgroup"])
        if not u_moodle_hash.empty:
            df = df.merge(u_moodle_hash, on="moodle_user_id", how="right")
        
        return df
    
    
    