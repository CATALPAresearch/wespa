import os
import pandas as pd

from .settings import *
from .util import prnt
from .util import print_all_output


class Extract_Degree:

    def __init__(self, semester, period_split_interval=0, save_output=True):
        self.semester = semester
        self.subset_until = 0
        self.period_split_interval = period_split_interval
        self.save_output=save_output

    def summarize_individual_level(self, author_relations, subset_until=0):
        """
        """
        self.subset_until = subset_until
        # Step 1: Summarize results per group and author
        author_relations['left_right'] = author_relations['left_neighbor'].astype(str) + '-' + author_relations['right_neighbor'].astype(str)
        author_relations['count'] = author_relations['count'].astype(float)
        author_relations['n'] = 1  # Equivalent to adding a count column for later summation
        
        # Group by group, author, and left_right to compute summarised statistics
        author_relations_summary = (
            author_relations
            .groupby(['moodle_group_id', 'moodle_user_id', 'left_right'], as_index=False)
            .agg(count_changesets=('n', 'sum'), char_count=('count', 'sum'))
        )

        # Extract left and right neighbors #FixMe? do we need this?
        author_relations_summary['left_right'] = author_relations_summary['left_right'].apply(lambda x: x if '-' in x else f"{x}-0")

        # Splitting left_right into separate columns safely
        split_values = author_relations_summary['left_right'].str.split('-', expand=True)
        author_relations_summary['left'] = split_values[0].astype('int') if len(split_values) > 0 else -1000
        author_relations_summary['right'] = split_values[1].astype('int').abs() if len(split_values) > 0 else -1000 

        # Return relevant columns
        res = author_relations_summary[['moodle_group_id', 'moodle_user_id', 'left', 'right', 'count_changesets', 'char_count']]
        res['until'] = self.subset_until
        res['moodle_user_id'] = res['moodle_user_id'].astype(int)
        self.save_data(res, 'author_relations_summary.csv')
        return res


    def extract_degree(self, author_relations_summary, subset_until=0):
        """
        """
        self.subset_until = subset_until
        # Step 1: Compute degrees
        degrees = author_relations_summary.copy()
        prnt('step 1:'+ str( degrees.shape))
        degrees['self'] = degrees.apply(
            lambda row: row['count_changesets'] if (
                (row['left'] == row['moodle_user_id'] and row['right'] == 0) or 
                (row['left'] == 0 and row['right'] == row['moodle_user_id']) or 
                (row['left'] == row['moodle_user_id'] and row['right'] == row['moodle_user_id']) or 
                (row['left'] == 0 and row['right'] == 0)) else 0, 
                axis=1
        )
        prnt('step 2:'+ str(degrees.shape))
        degrees['outgoing'] = degrees.apply(
            lambda row: row['count_changesets'] if (row['left'] != row['moodle_user_id'] or row['right'] != row['moodle_user_id']) else 0, 
            axis=1
        )
        prnt('step 3:'+ str(degrees.shape))
        degrees['self_chars'] = degrees.apply(
            lambda row: row['char_count'] if (
                (row['left'] == row['moodle_user_id'] and row['right'] == 0) or 
                (row['left'] == 0 and row['right'] == row['moodle_user_id']) or 
                (row['left'] == row['moodle_user_id'] and row['right'] == row['moodle_user_id']) or 
                (row['left'] == 0 and row['right'] == 0)) else 0, 
                axis=1
        )
        prnt('step 4:'+ str( degrees.shape))
        degrees['outgoing_chars'] = degrees.apply(
            lambda row: row['char_count'] if (row['left'] != row['moodle_user_id'] or row['right'] != row['moodle_user_id']) else 0, 
            axis=1
        )

        # Summarizing by author
        degrees = degrees.groupby('moodle_user_id').agg(
            selfdegree_count=('self', 'sum'),
            outdegree_count=('outgoing', 'sum'),
            selfdegree_chars=('self_chars', 'sum'),
            outdegree_chars=('outgoing_chars', 'sum')
        ).reset_index()
        prnt('step 5:'+ str( degrees.shape))
        degrees['total_count'] = degrees['selfdegree_count'] + degrees['outdegree_count']
        degrees['total_chars'] = degrees['selfdegree_chars'] + degrees['outdegree_chars']
        prnt('step 6:'+ str( degrees.shape))

        # Step 2: Compute indegree from left
        author_relations_summary_copy = author_relations_summary.copy()
        indegree_left = author_relations_summary_copy[author_relations_summary_copy['left'] != 0].groupby('left').agg(
            indegree_l=('count_changesets', 'sum'),
            indegree_l_chars=('char_count', 'sum')
        ).reset_index().rename(columns={'left': 'moodle_user_id'})
        prnt('step 7:'+ str( indegree_left.shape))

        # Step 3: Compute indegree from right
        indegree_right = author_relations_summary_copy[author_relations_summary_copy['right'] != 0].groupby('right').agg(
            indegree_r=('count_changesets', 'sum'),
            indegree_r_chars=('char_count', 'sum')
        ).reset_index().rename(columns={'right': 'moodle_user_id'})
        prnt('step 8:'+ str( indegree_right.shape))
        prnt(indegree_right['indegree_r'])
        # Step 4: Combine left and right indegrees
        indegree = pd.merge(indegree_left, indegree_right, on='moodle_user_id', how='outer').fillna(0)
        #print(indegree['indegree_r'])
        indegree['indegree_count'] = indegree['indegree_l'] + indegree['indegree_r']
        indegree['indegree_chars'] = indegree['indegree_l_chars'] + indegree['indegree_r_chars']
        prnt('step 9:'+ str( indegree.shape))
        # Step 5: Combine all degrees data
        #author_degrees = pd.merge(degrees, indegree, how="outer", on=["moodle_user_id", "moodle_user_id"])
        author_degrees = pd.merge(degrees, indegree, on='moodle_user_id', how='outer').fillna(0)
        author_degrees = author_degrees[[
            'moodle_user_id', 
            'outdegree_count', 
            'indegree_count', 
            'selfdegree_count',
            'outdegree_chars', 
            'indegree_chars', 
            'selfdegree_chars'
            ]]
        author_degrees = author_degrees.sort_values(by=['selfdegree_count', 'outdegree_count'], ascending=[True, False])
        prnt('step 10:'+ str( author_degrees.shape))
        
        # Display sorted results
        author_degrees.sort_values(by='outdegree_count', ascending=False)
        prnt('step 11:'+ str(author_degrees.shape))
        
        author_degrees['until'] = self.subset_until
        #print(author_degrees)
        return author_degrees
    

    def map_to_group(self, df_textchanges, author_degrees, subset_until=0):
        self.subset_until = subset_until
        """
        """
        #print('map')
        #print(df_textchanges.shape, df_textchanges[3]['moodle_user_id'])
        #print(author_degrees.shape, author_degrees[3]['moodle_user_id'])
        # Step 1: Create a mapping of authors to groups
        group_author_mapping = df_textchanges[['moodle_user_id', 'moodle_group_id']].drop_duplicates()

        #author_degrees['moodle_user_id'] = author_degrees['moodle_user_id'].astype(int)
        #group_author_mapping['moodle_user_id'] = group_author_mapping['moodle_user_id'].astype(int)
        # Step 2: Merge author degrees with group mapping
        authordegrees = author_degrees.merge(
            group_author_mapping, 
            left_on='moodle_user_id', 
            right_on='moodle_user_id', 
            how='right'
        )
        prnt('step 1' + str(authordegrees.shape))
        # Step 3: Rename and reorder columns
        authordegrees = authordegrees.rename(columns={'moodle_group_id': 'moodle_group_id'})
        authordegrees = authordegrees[['moodle_user_id', 'moodle_group_id'] + [col for col in authordegrees.columns if col not in ['moodle_user_id', 'moodle_group_id', 'moodle_user_id']]]
        prnt('step 2' + str(authordegrees.shape))
        # Step 4: Filter out missing values # FixMe
        authordegrees = authordegrees.dropna(subset=['outdegree_count', 'selfdegree_count'])
        prnt('step 3' + str(authordegrees.shape))
        # Step 5: Sort the results
        authordegrees = authordegrees.sort_values(by='outdegree_count', ascending=False)
        prnt('step 4' + str(authordegrees.shape))
        authordegrees['until'] = self.subset_until
        authordegrees['moodle_user_id'] = authordegrees['moodle_user_id'].astype(int)
        self.save_data(authordegrees, 'author-degrees.csv')
        return authordegrees
    

    def save_data(self, df, filename):
        """ Save data to CSV file"""
        if self.save_output == False:
            return
        
        file_path = f'{output_path}/{project_name}-{self.semester}-etherpad-06-{filename}' # {self.period_split_interval
        df['semester'] = self.semester
        df.to_csv(
            file_path, 
            index=False,
            quotechar='"',
            header = not os.path.exists(file_path), # False if self.subset_until != 0 else True,
            mode = 'a' if self.subset_until != 0 else 'w'
            )  

