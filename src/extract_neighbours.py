import pandas as pd
import numpy as np
import time
from datetime import datetime

from .settings import *
from .util import prnt
from .util import print_all_output


class Extract_Neighbours:
    def __init__(self,semester,):
        self.semester = semester
        
    def extract_neighbours(self, df_textchanges):
        # Step 1 was R parallelization
        df_textchanges_short = df_textchanges

        # Step 2: Create an empty DataFrame with required columns
        author_relations = pd.DataFrame(columns=['group', 'pad', 'author', 'left_neighbor', 'right_neighbor', 'count'])

        # Get unique pad IDs from df_textchanges_short
        df_textchanges['moodle_pad_id_'] = df_textchanges['moodle_pad_id'].str.split('$', expand=True)[0]
        pads = df_textchanges_short['moodle_pad_id_'].unique()
        pads_length = len(pads)
        
        # Step 3: Determine neighbors
        max_position = 0
        for i, pad in enumerate(pads):
            start_time = time.time()
            prnt('Process pad ' + str(i) + '/' + str(pads_length) + ' Position range from 0 to ' + str(max_position) + '__rows '+str(len(df_textchanges_short[df_textchanges_short['moodle_pad_id'] == pad])))
            
            # Step 4: Prepare array for storing the author of each character
            max_position = df_textchanges_short.loc[df_textchanges_short['moodle_pad_id_'] == pad, 'position'].max()
            max_change_length = df_textchanges_short.loc[df_textchanges_short['moodle_pad_id_'] == pad, 'textchange'].max()
            max_position = abs(max_position) + abs(max_change_length) +100
            char_authors = np.array(np.zeros(max_position, dtype=int))  # Initialize a list of zeros
            
            # Step 5: Identifiy neighbours of the position of each changeset
            pad_changes = []
            for _, change in df_textchanges_short[df_textchanges_short['moodle_pad_id_'] == pad].iterrows():
                text_length = int(change['textchange'])
                author = change['moodle_author_id']
                pad_id = change['moodle_pad_id_']
                group_id = change['moodle_group_id']
                start_pos = int(change['position'])
                end_pos = start_pos + text_length if text_length >= 0 else start_pos #NewFix

                # Ensure start_pos is within bounds
                start_pos = max(0, min(start_pos, max_position - 1))
                if end_pos < start_pos:
                    pass
                    print('Error Pad: start_pos>end_pos ' + str(i) + ' Positions ' + str(start_pos) +' :: '+ str(end_pos) + '__' + str(len(char_authors)) +'___'+ change['textedit_changeset'])
                
                # Determine neighbors
                if start_pos + 1 > max_position:
                    print('Error Pad: start_pos+1 > max_position '+str(start_pos)+' +1 < '+str(max_position)+'  '+str(int(start_pos) + 1 <= int(max_position)))
                left_neighbor = char_authors[start_pos] if start_pos >= 0 else -100
                right_neighbor = char_authors[start_pos + 1] if int(start_pos) + 1 <= int(max_position) else -200
                
                # Update char_authors list
                if text_length > 0: # add author fields for added chars
                    char_authors = np.concatenate([
                        char_authors[0:start_pos-1], 
                        [int(author)] * text_length, 
                        char_authors[start_pos:]
                        ])
                elif text_length < 0: # Remove author information for removed chars
                    char_authors = np.concatenate([
                        char_authors[:start_pos-text_length], 
                        char_authors[start_pos:]
                        ])

                # Append result to pad_changes list
                pad_changes.append([group_id, pad_id, author, left_neighbor, right_neighbor, text_length])

            # Convert to DataFrame and add to author_relations
            pad_changes_df = pd.DataFrame(pad_changes, columns=['group', 'pad', 'author', 'left_neighbor', 'right_neighbor', 'count'])
            author_relations = pd.concat([author_relations, pad_changes_df], ignore_index=True)

            #print(f"{i+1}/{pads_length} {pad} {len(df_textchanges_short[df_textchanges_short['moodle_pad_id_'] == pad])} {start_time - time.time()} sec")

        self.save_data(author_relations, 'author-relations.csv')
        return author_relations
        

    def test_missing_neighbors(self, df_textchanges):
        """Test"""
        #TODO
        author_relations = self.extract_neighbours(df_textchanges)
        author_relations[author_relations['left_neighbor']==-100]
        author_relations[author_relations['right_neighbor']==-200]

    
    def save_data(self, df, filename):
        """ Save data to CSV file"""
        df.to_csv(
            f'{output_path}/{project_name}-{self.semester}-03-{filename}', 
            index=False,
            quotechar='"'
            )        
