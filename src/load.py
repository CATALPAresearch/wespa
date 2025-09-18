import os
import pandas as pd
import numpy as np

from .settings import *
from .util import prnt

class Load:

    def __init__(self, semester, dump='dump20240826', data_path='./data'):
        self.semester = semester
        self.dump = dump
        self.data_path = data_path
        self.period_1 = get_period(semester, period_1_arr)
        self.period_2 = get_period(semester, period_2_arr)
        self.period_3 = get_period(semester, period_3_arr)

        self.df_textedits = None
        self.df_chats = None
        self.df_comment_replies = None
        self.df_comments = None
        self.df_scrolls = None
    
    def load_data(self, filename):
        return pd.read_csv(f'{self.data_path}/{self.dump}/{filename}')
    

    def preprocess(self, df, activity, filter_group=None, filter_weeks=None):
        """ Convert column format and filter data """
        if filter_group != None:# and type(filter_group).__name__ in ('list', 'tuple'):
            prnt('filter-'+activity)
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
        
        # remove teachers and empty author fields
        df = df[~df['authorid'].isin(teacher_ids)]
        df = df[df['authorid'] != '']
        df = df[~df['authorid'].isnull()]

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
        df["type"] = "comments"
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-comments.csv')
        self.save_data(df_summary, 'data-summary-comments.csv')
        self.df_comments = df
    
    def process_comment_replies(self, filter_group=None, filter_weeks=None):
        print(f'Loading comment replies data from CSV')
        df = self.preprocess(self.load_data('pad_comment_reply.csv'), 'commentreply', filter_group, filter_weeks)
        df["type"] = "comment_replies"
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-comment-replies.csv')
        self.save_data(df_summary, 'data-summary-comment-replies.csv')
        self.df_comment_replies = df
    
    def process_chats(self, filter_group=None, filter_weeks=None):
        print(f'Loading chat data from CSV')
        df = self.preprocess(self.load_data('pad_chat.csv'), 'chat', filter_group, filter_weeks)
        df["type"] = "chats"
        df_summary = self.summarize(df, 'text_length')
        self.save_data(df, 'data-chats.csv')
        self.save_data(df_summary, 'data-summary-chats.csv')
        self.df_chats = df
    

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
        df["type"] = "scrolls"
        
        df_summary = df.groupby(['authorid', 'period']).agg(
            sum_scroll_events=('authorid', 'count'),
            sum_scroll_distance=('distance', 'sum'),
            sum_scroll_distance_up=('distance_up', 'sum'),
            sum_scroll_distance_down=('distance_down', 'sum')
        ).reset_index()
        
        self.save_data(df, 'data-scrolls.csv')
        self.save_data(df_summary, 'data-summary-scrolls.csv')
        self.df_scrolls = df
    

    def process_textedits(self, filter_group=None, filter_weeks=None):
        print(f'Loading textedit data from CSV')
        df = self.preprocess(self.load_data('pad_commit.csv'), 'textedit', filter_group, filter_weeks)
        # Step 2: Rename columns for consistency
        df = df.rename(columns={
            'id': 'id',
            'userid': 'moodle_user_id',
            'groupid': 'moodle_group_id',
            'padid': 'moodle_pad_id',
            'taskid': 'moodle_task_id',
            'text': 'textedit_changeset',
            'rev': 'textedit_rev',
            'type': 'type'
        })
        df["type"] = "textedits"
        
        # Step 6: Hash Moodle user IDs (if enabled)
        if hashit:
            df = self.hashUsers(df)
        
        # Step 8: Remove groups with only one active member
        active_groups = df.groupby("moodle_group_id")['moodle_user_id'].nunique()
        df = df[df['moodle_group_id'].isin(active_groups[active_groups > 1].index)]
        df["id"] = df.index
        # Step 9: Select only required columns
        selected_columns = [
            'id', 'moodle_user_id', 'moodle_group_id', 'moodle_pad_id',
            'textedit_changeset', 'timestamp', 'week', 'period', 'type'
        ]
        df = df[selected_columns]

        self.save_data(df, 'data-textedits.csv')
        self.df_textedits = df


    def create_user_id_mappings(self):
        """
        """
        print(f'Create list of all users and groups')
        users_comments = self.df_comments[['authorid', 'groupid', 'userid']]
        users_comment_replies = self.df_comment_replies[['authorid', 'groupid', 'userid']]
        users_chat = self.df_chats[['authorid', 'groupid', 'userid']]
        users_scrolls = self.df_scrolls[['authorid', 'groupid', 'userid']]
        users_all = pd.concat([users_comments, users_comment_replies, users_chat, users_scrolls], ignore_index=True)
        users_all = users_all[['authorid', 'groupid', 'userid']].drop_duplicates()
        users_all['userid'] = users_all['userid'].fillna(0).astype(int)
        users_all = users_all.rename(columns={
            'authorid': 'etherpad_user_id',
            'userid': 'moodle_user_id',
            'groupid': 'etherpad_group_id'
        })

        # match with moodle_group_id
        users_textedits = self.df_textedits[['moodle_user_id', 'moodle_group_id']]

        # join moodle_user_id in users_all with moodle_user_id in users_textedits
        print('length before ' + str(len(users_all.index)))
        users_merged = users_textedits.merge(
            users_all,
            left_on='moodle_user_id', 
            right_on='moodle_user_id', 
            how='right',
            suffixes=('', '_right') 
        )
        users_merged = users_merged[users_merged['moodle_user_id'] == users_merged['moodle_user_id']] 
        users_merged = users_merged[['moodle_user_id', 'moodle_group_id', 'etherpad_user_id', 'moodle_user_id']].drop_duplicates()

        print('length after ' + str(len(users_merged.index)))

        self.save_data(users_merged, 'data-users.csv')
        
        
    def get_data(self):
        """
        Get textedit data
        """
        return self.df_textedits    


    def save_data(self, df, filename):
        """ Save data to CSV file"""
        df['semester'] = self.semester
        df.to_csv(
            f'{output_path}/{project_name}-{self.semester}-etherpad-01-{filename}', 
            index=False,
            quotechar='"'
            )
    

    def run(self, filter_group=None, filter_weeks=None):
        """ Run all preprocessing tasks """
        self.process_textedits(filter_group, filter_weeks)
        self.process_comments(filter_group, filter_weeks)
        self.process_comment_replies(filter_group, filter_weeks)
        self.process_chats(filter_group, filter_weeks)
        self.process_scrolls(filter_group, filter_weeks)
        self.create_user_id_mappings()

    def hashUsers(self, df):
        """ TODO hash """
        def load_mapping(file_path, cols):
            if os.path.exists(file_path):
                return pd.read_csv(file_path, delimiter=';', skiprows=1, names=cols, encoding='utf-8')
            else:
                print(f"Error: File {file_path} does not exist.")
                return pd.DataFrame()

        moodle_etherpad_file = f'../data/{self.semester}_user_mapping_moodle_etherpad.csv'
        etherpad_hash_file = f'../data/{self.semester}_user_mapping_etherpad_hash.csv'
        moodle_hash_file = f'../data/{self.semester}_user_mapping_moodle_hash.csv'

        u_moodle_etherpad = load_mapping(moodle_etherpad_file, ["moodle_user_id", "etherpad_user_id"])
        if not u_moodle_etherpad.empty:
            df = df.merge(u_moodle_etherpad, left_on="moodle_user_id", right_on="moodle_user_id", how="right")
        
        u_etherpad_hash = load_mapping(etherpad_hash_file, ["etherpad_user_id", "hashuser", "hashgroup", "groupcat"])
        if not u_etherpad_hash.empty:
            df = df.merge(u_etherpad_hash, on="etherpad_user_id", how="right")
        
        u_moodle_hash = load_mapping(moodle_hash_file, ["moodle_user_id", "hashuser", "hashgroup"])
        if not u_moodle_hash.empty:
            df = df.merge(u_moodle_hash, on="moodle_user_id", how="right")
        
        return df
    
    
    