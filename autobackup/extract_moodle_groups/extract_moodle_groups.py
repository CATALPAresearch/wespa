import traceback
import sys
import requests
import re
from bs4 import BeautifulSoup
import os
import json
import argparse

def extract_moodle_groups(mdl_host, mdl_courseid, mdl_username, mdl_password, format='file'):
    if not all([mdl_host, mdl_courseid, mdl_username, mdl_password]):
        print('Error: Not all required parameters have been provided')
        return
    try:
        groups = []
        session = requests.Session()
        req = session.get(mdl_host + "/login/index.php")
        pattern_auth = r'<input type="hidden" name="logintoken" value="\w{32}">'
        token = re.findall(pattern_auth, req.text)
        token = re.findall(r'\w{32}', token[0])[0]    
        login = { 'anchor': '', 'logintoken': token, 'username': mdl_username, 'password': mdl_password, 'rememberusername': 1}
        session.post(mdl_host + "/login/index.php", data=login)
        # We want to get the group list
        d = session.get(mdl_host + "/group/index.php?id="+str(mdl_courseid))
        if d.status_code != 200:
            raise Exception('Could not receive data.')
        soup = BeautifulSoup(d.content, 'html.parser')
        element = soup.find(id='groups')
        for child in element.children:
            v = child.getText()
            c = str(child)        
            f = c.find('value="')
            if f <= 0:
                continue
            id = (int)(c[f:].split('"')[1])
            groups.append({"id": id, "members":[], 'name': v})
        for group in groups:
            p = mdl_host + '/group/members.php?group='+str(group['id'])
            r = session.get(p)
            if r.status_code != 200:
                raise Exception('Could not fetch group '+str(group['id']))
            soup = BeautifulSoup(r.content, 'html.parser')
            elements = soup.find(id='removeselect').find_all('option')
            ids = []
            for element in elements:
                c = str(element)
                f = c.find('value="')
                if f <= 0:
                    continue
                id = (int)(c[f:].split('"')[1])
                ids.append(id)
            group['members'] = ids
        if format=='file':
            f = open(os.path.join(os.getcwd(), 'groups.json'), 'w')
            json.dump(groups, f)
            return
        else:
            return json
    except Exception:
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Moodle processing script')
    
    parser.add_argument('--host', required=True, help='Moodle host URL')
    parser.add_argument('--courseid', required=True, help='Course ID')
    parser.add_argument('--username', required=True, help='Moodle username')
    parser.add_argument('--password', required=True, help='Moodle password')
    
    args = parser.parse_args()
    
    extract_moodle_groups(args.host, args.courseid, args.username, args.password)