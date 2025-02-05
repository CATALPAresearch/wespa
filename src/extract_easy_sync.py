import pandas as pd
import re

from .settings import *
from .util import prnt
from .util import print_all_output


class Extract_Easy_Sync:

    # settings
    reconstrcut_text = False
    
    def __init__(self,semester):
        self.semester = semester

    def extract_changeset(self, changeset, feature='all'):
        """
        Extract changeset features
        """
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

        chanched_text = ''
        clean_source = ''
        payload_split = str(source_code).split('$')
        print('problem space ',payload_split)
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

        #prefix = re.findall(r"\:\d+[\>\=\<]\d+", clean10_source)[0]
        #clean_changes = clean10_source.replace(prefix, '')
        #changes = re.split(r"[\+\-]", clean_changes)

        # convert base36 to int
        operators = [':','+','-','*','=','>','<','|']
        change_operators = ['+','-']
        tmp = ''
        clean10_source = ''
        for char in clean_source:
            if char in operators:
                if tmp != '':
                    clean10_source = clean10_source + str(int(tmp, 36)) + char
                    tmp = ''
            else:
                tmp = tmp + char
        clean10_source = ':' + clean10_source + str(tmp) #FixMe ... removed conversion of tmp to base 36
        prnt('source converted from base36:\t\t' + clean10_source)

        # split
        changes = []
        tmp2 = ''
        prefix = re.findall(r"\:\d+[\>\=\<]\d+", clean10_source)
        if len(prefix)>0:
            prefix = prefix[0]
        else:
            prefix = ''
        clean10_source_ = clean10_source.replace(prefix, '')
        i = 0
        while i < len(clean10_source_):
            tmp2 = tmp2 + clean10_source_[i]
            if clean10_source_[i] in change_operators:
                changes.append(tmp2 + clean10_source_[i+1])
                i = i+1
                tmp2 = ''
            i = i+1
    
        # Text length
        source_text_length = int(re.findall(r":\d+", clean10_source)[0].replace(':',''))
        prnt('source text length:\t\t' + str(source_text_length))
        #FixMe original length + changes
        changed_text_length = int(re.findall(r"[\>\<]\d+", clean10_source)[0].replace('>','').replace('<',''))
        if "<" in clean10_source:
            changed_text_length = changed_text_length * -1
            pass
        # prnt(str(source_text_length+changed_text_length) + '=' + str(source_text_length) + '-' + str(changed_text_length))
        prnt('final text length:\t\t' + str(changed_text_length))

        # split individual changes of the changeset
        position = 1
        line_number = 1
        chars_added = 0
        chars_removed = 0
        chars_kept = 0
        res = []
        for change in changes:
            prnt('-------------------')
            prnt(change)
            # update position from previous change operation
            position = position + chars_kept + chars_added - chars_removed 

            # character changes
            chars_added = re.findall(r"\+\d+", change)
            chars_added = sum([int(char.replace('+', '')) for char in chars_added])
            chars_removed = re.findall(r"\-\d+", change)
            chars_removed = sum([int(char.replace('-', '')) for char in chars_removed])
            pattern = r"(?:^|[^|])\d*=(\d+)"
            chars_kept = re.findall(pattern, change)
            chars_kept = sum([int(char.replace('=', '')) for char in chars_kept])
            
            prnt('chars added:\t\t' + str(chars_added))
            prnt('chars removed:\t\t' + str(chars_removed))
            prnt('chars kept:\t\t' + str(chars_kept))
            
            # line changes
            lines_added = re.findall(r"\|\d+\+\d+", change)
            number_of_lines_added = sum([int(lines.split('+')[0].replace('|','')) for lines in lines_added])
            number_of_chars_on_added_line = sum([int(lines.split('+')[1]) for lines in lines_added])
            lines_removed = re.findall(r"\|\d+\-\d+", change)
            number_of_lines_removed = sum([int(lines.split('-')[0].replace('|','')) for lines in lines_removed])
            number_of_chars_on_removed_line = sum([int(lines.split('-')[1]) for lines in lines_removed])
            lines_kept = re.findall(r"\|\d+\=\d+", change)
            number_of_lines_kept = sum([int(lines.split('=')[0].replace('|','')) for lines in lines_kept])
            number_of_chars_on_kept_line = sum([int(lines.split('=')[1]) for lines in lines_kept])
            prnt('lines added:\t\t' + str(number_of_lines_added))
            prnt('chars on added lines:\t' + str(number_of_chars_on_added_line))
            prnt('lines removed:\t\t' + str(number_of_lines_removed))
            prnt('chars on removed lines:\t' + str(number_of_chars_on_removed_line))
            prnt('lines kept:\t\t' + str(number_of_lines_kept))
            prnt('chars on kept lines:\t' + str(number_of_chars_on_kept_line))
            
            # formatting
            formatting_operations = re.findall(r"\*\d+", change)
            formatting_operations = [fo.replace('*', '') for fo in formatting_operations]
            formatting_operations = [int(fo) for fo in formatting_operations]
            number_of_formatting_operations = len(formatting_operations)
            prnt('formatting operations:\t' + str(formatting_operations))
            prnt('number_of_formatting_operations:' + str(number_of_formatting_operations))   
            
            # positioning
            position = position + chars_kept
            line_number = line_number + number_of_lines_kept + number_of_lines_added - number_of_lines_removed
            prnt('position:\t\t' + str(position))
            prnt('line_number:\t\t' + str(line_number))

            # reconstruct text
            if self.reconstrcut_text==True:
                if number_of_lines_added > 0:
                    self.reconstruct_text(number_of_chars_on_added_line, 'add_lines', number_of_lines_added)
                if number_of_lines_removed > 0:
                    self.reconstruct_text(number_of_chars_on_removed_line, 'remove_lines', number_of_lines_removed)
                if chars_added > 0:
                    self.reconstruct_text(position, 'add_chars', chanched_text)
                if chars_removed > 0:
                    self.reconstruct_text(position, 'remove_chars', chars_removed)
            

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


    def reconstruct_text(self, pos, operation, value):
        """
        Reconstruct text
        """
        match operation:
            case 'add_chars':
                self.ttext = self.ttext[0:pos] + value + self.ttext[pos:]
            case 'remove_chars':
                self.ttext = self.ttext[0:pos] + self.ttext[pos+value:]
            case 'add_line':
                self.ttext = self.ttext[0:pos] + ('\n' * value) + self.ttext[pos:]
                pass
            case 'remove_line':
                tail = self.ttext[pos:].replace('\n'*value, '', 1)
                self.ttext = self.ttext[0:pos] + tail
                pass
        return self.ttext


    def extract_easy_sync(self, df_textedit):
        """
        Augment the textchange dataframe with extracted information of the changeset
        """
        print_all_output = False
        self.reconstrcut_text = False
        df_textchanges = (
            #df_textedit #.head()
            df_textedit.groupby('moodle_pad_id').first() # to test text reconstrcution
            .groupby('moodle_pad_id', group_keys=False)  # Group by pad ID
            .apply(
                lambda group: group.assign(
                    week=group['timestamp'].apply(
                        lambda ts: pd.to_datetime(ts, unit='s').strftime("%y-%U")
                    ),
                    # Apply extract_changeset once and expand results into multiple columns
                    **group['textedit_changeset'].apply(self.extract_changeset).apply(pd.Series),
                    # Compute dependent columns based on extracted metrics
                    textedit_charsadded=lambda g: g['sourceTextLength'] + g['textchange']
                )
            )
            .reset_index(drop=False)
        )
        self.save_data(df_textchanges, 'textchanges.csv')
        return df_textchanges
    

    def save_data(self, df, filename):
        """ Save data to CSV file"""
        df.to_csv(
            f'{output_path}/{project_name}-{self.semester}-02-{filename}', 
            index=False,
            quotechar='"'
            )






