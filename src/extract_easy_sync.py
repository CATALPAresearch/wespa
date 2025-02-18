import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta
import math

from .settings import *
from .util import prnt
from .util import print_all_output



class Extract_Easy_Sync:
    global print_all_output
    
    def __init__(self,semester, time_breaks, period_split_interval=0):
        self.semester = semester
        self.time_breaks = time_breaks
        self.period_split_interval = period_split_interval
        self.ttext = ''
        self.tmp_pad_id = ''
        
        self.tmp_timestamp = datetime(2000, 1, 1).timestamp()
        self.current_time_threshold = datetime(2000, 1, 1).timestamp()
        self.observation_timestamps = []
        self.is_reconstructing_text = False
        self.add_chars_done = True
        print_all_output = True
        
        

    def convertChangsetBase36ToInt(self, clean_source):
        """Converts the Base36 numbers in the changeset into Base10 integers"""
        operators = [':','+','-','*','=','>','<','|']
        tmp = ''
        clean10_source = ''
        for char in clean_source:
            if char in operators:
                if tmp != '':
                    clean10_source = clean10_source + str(int(tmp, 36)) + char
                    tmp = ''
            else:
                tmp = tmp + char
        clean10_source = ':' + clean10_source + str(int(tmp, 36)) 
        prnt('source converted from base36:\t\t' + clean10_source)
        return clean10_source


    def extract_changeset(self, changeset, feature='all', timestamp=None, group_id=None, pad_id=None):
        """
        Extract changeset features
        """
        self.add_chars_done = False
        # reset
        if pad_id != self.tmp_pad_id:
            print('New Pad: ' + str(pad_id))
            self.ttext = ''
        self.tmp_pad_id = pad_id
        chars_added_total = 0
        chars_removed_total = 0
        chars_kept_total = 0
        number_of_lines_added_total = 0
        number_of_chars_on_added_line_total = 0
        number_of_lines_removed_total = 0
        number_of_chars_on_removed_line_total = 0
        number_of_lines_kept_total = 0
        number_of_chars_on_kept_line_total = 0
        formatting_operations_total = []
        number_of_formatting_operations_total = 0

        source_code = changeset
        prnt('source:\t\t' + str(source_code))

        # Step 1: Seperate payload from operations
        chanched_text = ''
        clean_source = ''
        payload_split = str(source_code).split('$')
        
        if len(payload_split)>1:
            chanched_text = payload_split[1].replace('$','')
            clean_source = payload_split[0].replace('$','')
        else:
            chanched_text = ''
            clean_source = payload_split.replace('$','')
        prnt('changed text:\t\t' + chanched_text)
        prnt('clean_source '+clean_source)

        init_symbol_split = str(clean_source).split("Z")
        if len(clean_source) > 1:
            clean_source = init_symbol_split[1]
        else:
            clean_source = str(clean_source).replace('Z','')

        
        # Step 2: Convert changeset base36 to int
        clean10_source = self.convertChangsetBase36ToInt(clean_source)
    
        
        # Step 3: Extract general parameters
        # Text length
        source_text_length = int(re.findall(r":\d+", clean10_source)[0].replace(':',''))
        prnt('source text length:\t\t' + str(source_text_length))
        #FixMe original length + changes
        changed_text_length = int(re.findall(r"[\>\<]\d+", clean10_source)[0].replace('>','').replace('<',''))
        if "<" in clean10_source:
            changed_text_length = changed_text_length * -1
        prnt('final text length:\t\t' + str(changed_text_length))

        # Step 4: Split operations of the changeset
        prefix = re.findall(r"\:\d+[\>\=\<]\d+", clean10_source)
        prefix = prefix[0] if len(prefix)>0 else ''
        clean10_source_ = clean10_source.replace(prefix, '')
        changes = re.split(r'(?=\|)', clean10_source_)
        prnt('base' + clean10_source_)
        prnt('splitted base: ' + '  --  '.join(changes))

        # Step 5: process individual changes of the changeset
        position = 0
        line_number = 1
        chars_added = 0
        chars_removed = 0
        chars_kept = 0
        tmp_chanched_text = chanched_text
        tmp_text = ''
        for change in changes:
            # update position from previous change operation
            #position = position + chars_kept + chars_added - chars_removed 

            # Step 1: formatting
            formatting_operations = re.findall(r"\*\d+", change)
            formatting_operations = [fo.replace('*', '') for fo in formatting_operations]
            formatting_operations = [int(fo) for fo in formatting_operations]
            number_of_formatting_operations = len(formatting_operations)
            # TODO: mapp particualr formatting operations
            # remove formating operations for further analysis
            change = re.sub(r"\*\d+", '', change)
            prnt('formatting operations:\t' + str(formatting_operations))
            prnt('number_of_formatting_operations:' + str(number_of_formatting_operations))   
            
            # Step 2: line changes
            # |L+N: Insert N characters from the source text, containing L newlines. The last character inserted MUST be a newline, but not the (new) document's final newline.
            lines_added = re.findall(r"\|\d+\+\d+", change)
            number_of_lines_added = sum([int(lines.split('+')[0].replace('|','')) for lines in lines_added])
            number_of_chars_on_added_line = sum([int(lines.split('+')[1]) for lines in lines_added])
            if number_of_lines_added > 0 and self.is_reconstructing_text==True:
                value = tmp_chanched_text[0:number_of_chars_on_added_line]
                tmp_chanched_text = tmp_chanched_text[number_of_chars_on_added_line:]
                self.ttext = self.ttext[0:position] + value + ('\n' * number_of_lines_added) + self.ttext[position:]
                position = position + number_of_chars_on_added_line

            # |L-N: Delete N characters from the source text, containing L newlines. The last character inserted MUST be a newline, but not the (old) docu- ment's nal newline.
            lines_removed = re.findall(r"\|\d+\-\d+", change)
            number_of_lines_removed = sum([int(lines.split('-')[0].replace('|','')) for lines in lines_removed])
            number_of_chars_on_removed_line = sum([int(lines.split('-')[1]) for lines in lines_removed])
            if number_of_lines_removed > 0 and self.is_reconstructing_text==True:
                #print('--',number_of_lines_removed,'--  ',self.ttext[position-4:position+number_of_chars_on_removed_line+number_of_lines_removed],'</end>')
                #self.ttext = self.ttext[0:position] + self.ttext[position+number_of_chars_on_removed_line:]
                #position = position - number_of_chars_on_removed_line
                #self.ttext = self.ttext[0:position] + self.ttext[position:]
                pass

            # |L=N: Keep N characters from the source text, containing L newlines. The last character kept MUST be a newline, and the final newline of the document is allowed.
            lines_kept = re.findall(r"\|\d+\=\d+", change)
            number_of_lines_kept = sum([int(lines.split('=')[0].replace('|','')) for lines in lines_kept])
            number_of_chars_on_kept_line = sum([int(lines.split('=')[1]) for lines in lines_kept])
            if number_of_lines_kept > 0 and self.is_reconstructing_text==True:
                position = position + number_of_chars_on_kept_line
                self.ttext = self.ttext[0:position] + ('\n' * number_of_lines_added) + self.ttext[position:]
                    
            # remove line operations before the next analysis step
            change = re.sub(r"\|\d+\+\d+", '', change)
            change = re.sub(r"\|\d+\-\d+", '', change)
            change = re.sub(r"\|\d+\=\d+", '', change)
            prnt('lines added:\t\t' + str(number_of_lines_added))
            prnt('chars on added lines:\t' + str(number_of_chars_on_added_line))
            prnt('lines removed:\t\t' + str(number_of_lines_removed))
            prnt('chars on removed lines:\t' + str(number_of_chars_on_removed_line))
            prnt('lines kept:\t\t' + str(number_of_lines_kept))
            prnt('chars on kept lines:\t' + str(number_of_chars_on_kept_line))
            
            
            # Step 3: character changes
            chars_added = re.findall(r"\+[\d]+", change) ## FixMe: only the first character is converted from base36
            chars_added = sum([int(char.replace('+', '')) for char in chars_added])
            if len(chanched_text) > 0 and self.is_reconstructing_text==True:
                value = tmp_chanched_text[0:chars_added]
                tmp_chanched_text = tmp_chanched_text[chars_added:]
                self.ttext = self.ttext[0:position] + value + self.ttext[position:]
                position = position + chars_added
            
            chars_removed = re.findall(r"\-\d+", change)
            chars_removed = sum([int(char.replace('-', '')) for char in chars_removed])
            if chars_removed > 0 and self.is_reconstructing_text==True:
                self.ttext = self.ttext[0:position] + self.ttext[position+chars_removed:]
                    
            chars_kept = re.findall(r"(?:^|[^|])\d*=(\d+)", change)
            chars_kept = sum([int(char.replace('=', '')) for char in chars_kept])
            if chars_kept > 0:
                position = position + chars_kept

            prnt('chars added:\t\t' + str(chars_added))
            prnt('chars removed:\t\t' + str(chars_removed))
            prnt('chars kept:\t\t' + str(chars_kept))
            
            # positioning # TODO
            #position = position + chars_kept
            line_number = line_number + number_of_lines_kept + number_of_lines_added - number_of_lines_removed
            prnt('position:\t\t' + str(position))
            prnt('line_number:\t\t' + str(line_number))

            
            # sum up
            chars_added_total = chars_added_total + chars_added
            chars_removed_total = chars_removed_total + chars_removed
            chars_kept_total = chars_kept_total + chars_kept
            number_of_lines_added_total = number_of_lines_added_total + number_of_lines_added
            number_of_chars_on_added_line_total = number_of_chars_on_added_line_total + number_of_chars_on_added_line
            number_of_lines_removed_total = number_of_lines_removed_total + number_of_lines_removed
            number_of_chars_on_removed_line_total = number_of_chars_on_removed_line_total + number_of_chars_on_removed_line
            number_of_lines_kept_total = number_of_lines_kept_total + number_of_lines_kept
            number_of_chars_on_kept_line_total = number_of_chars_on_kept_line_total + number_of_chars_on_kept_line
            formatting_operations_total = formatting_operations_total + formatting_operations
            number_of_formatting_operations_total = number_of_formatting_operations_total + number_of_formatting_operations

        # save current text
        if self.is_reconstructing_text==True:
            self.save_reconstructed_text(timestamp, group_id, pad_id)
            pass    
            
        # return values
        match feature:
            case 'everything':
                return {
                    "source_code": source_code,
                    "clean_source": clean_source,
                    "clean_sourceBase10": clean10_source,
                    "source_text_length": source_text_length,
                    "changed_text": chanched_text,
                    "changed_text_length": changed_text_length,
                    "chars_added": chars_added_total,
                    "chars_removed": chars_removed_total,
                    "chars_kept": chars_kept_total,
                    "number_of_lines_added": number_of_lines_added_total,
                    "number_of_chars_on_added_line": number_of_chars_on_added_line_total,
                    "number_of_lines_removed": number_of_lines_removed_total,
                    "number_of_chars_on_removed_line": number_of_chars_on_removed_line_total,
                    "number_of_lines_kept": number_of_lines_kept_total,
                    "number_of_chars_on_kept_line": number_of_chars_on_kept_line_total,
                    "formatting_operations": formatting_operations_total,
                    "number_of_formatting_operations": number_of_formatting_operations_total,
                    'position': position,
                    'line_number': line_number,
                }
            case 'all': # = default
                return  {
                    "changeset": clean10_source,
                    "sourceTextLength": source_text_length,
                    "textchange": changed_text_length,
                    "numberOfAddedNewLines": number_of_lines_added_total,
                    "numberOfRemovedNewLines": number_of_lines_removed_total,
                    'position': position,
                    'line_number': line_number,
                }

    
    def save_reconstructed_text(self, timestamp, group_id, pad_id):
        """
        Save a snapshot of the reconstrcuted text 
        """
        tmp_timebreak = self.time_breaks[
            #self.time_breaks['moodle_group_id']== group_id & 
            self.time_breaks['moodle_pad_id']== pad_id
            ]
        if tmp_timebreak[tmp_timebreak['timestamp']==timestamp].shape[0] > 0:
            self.tmp_timestamp = timestamp
            pad_name = str(pad_id).replace('$','xxxx')
            file_path = f'{output_path}text/{project_name}-{self.semester}-{group_id}-{pad_name}-{self.period_split_interval}-{math.floor(timestamp)}.txt'
            with open(file_path, 'w') as f:
                f.write(self.ttext)
        

        """
            rows = []
            rows.append({
                "group_id": group_id, 
                "pad_id": pad_id, 
                "timestamp": timestamp, 
                "current_time_threshold": self.current_time_threshold, 
                "text": self.ttext  # Consider compressing this
            })
            tmp_df = pd.DataFrame(rows)
            tmp_df.to_csv(
                file_path, 
                index=False,
                quotechar='"',
                mode='a', 
                header = not os.path.exists(file_path)
            )
            """


    def generate_observation_times(self, start, end, period_split_interval):
        """Generates an array of observation times within a time range for a given delta of hours, days or weeks"""
        self.tmp_timestamp = start.timestamp()
        self.current_time_threshold = start.timestamp()
        start_date = start.timestamp()        
        end_date = end.timestamp()

        match period_split_interval:
            case "hours":
                timestamps = np.array([(start_date + timedelta(hours=i)).timestamp() for i in range((end - start).days*24 + 1)])
            case "days":
                timestamps = np.array([(start + timedelta(days=i)).timestamp() for i in range((end - start).days + 1)])
            case "weeks":
                timestamps = np.array([(start + timedelta(weeks=i)).timestamp() for i in range(math.floor((end-start).days/7) + 1)])

        self.observation_timestamps = timestamps
        
        return self.observation_timestamps
        

    def readbale_observation_times(self, unix_timestamp_arr):
        """Util function """
        readable_dates = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in unix_timestamp_arr]
        dfx = pd.DataFrame({'Timestamp': unix_timestamp_arr})
        dfx['Readable Date'] = pd.to_datetime(dfx['Timestamp'], unit='s')
        return dfx


    def extract_easy_sync(self, df_textedit):
        """
        Augment the textchange dataframe with extracted information of the changeset
        """
        df_textchanges = (
            #df_textedit #.head()
            #df_textedit.groupby('moodle_pad_id').first() # to test text reconstrcution
            df_textedit
            .groupby('moodle_pad_id', group_keys=False)  # Group by pad ID
            .apply(
                lambda group: group.assign(
                    week=group['timestamp'].apply(
                        lambda ts: pd.to_datetime(ts, unit='s').strftime("%y-%U")
                    ),
                    **group.apply(lambda row: 
                        self.extract_changeset(
                            row['textedit_changeset'], 
                            timestamp=row['timestamp'], 
                            group_id=row['moodle_group_id'], 
                            pad_id=str(row['moodle_pad_id'])
                            #pad_id=row[3]
                        ), axis=1).apply(pd.Series),
                    textedit_charsadded=lambda g: g['sourceTextLength'] + g['textchange'] # FixMe: do we need this?
                )
            )
            .reset_index(drop=True)
        )
        self.save_data(df_textchanges, 'textchanges.csv')
        return df_textchanges
    
    

    def save_data(self, df, filename):
        """ Save data to CSV file"""
        file_path = f'{output_path}{project_name}-{self.semester}-02-{filename}'
        df.to_csv(
            file_path, 
            index=False,
            quotechar='"',
            header = not os.path.exists(file_path)
            )






