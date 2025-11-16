import psycopg2

conn = psycopg2.connect(
    user="marc",
    password="password",
    host="localhost",
    port="8100",
    database="analytics" 
)
cursor = conn.cursor()

mods = [10863, 6280, 21, 8846, 8847, 3116, 7, 9, 10986, 8834, 6, 5, 13191, 13208, 13161, 13131, 10986, 3340, 13189, 13107, 13189]
mods_set = set(mods)
tables = ['pad_chat', 'pad_chat_scrolling', 'pad_chat_visibility', 'pad_comment', 'pad_comment_reply', 'pad_commit', 'pad_scrolling', 'pad_session', 'pad_visibility']

for modId in mods_set:
    for table in tables:
        query = f"UPDATE {table} SET moderator = true WHERE userid = %s"        
        cursor.execute(query, (modId,))
conn.commit()

print('end')