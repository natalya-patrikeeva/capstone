import sqlite3
from HTMLParser import HTMLParser
import itertools
from time import time

t0 = time()

conn = sqlite3.connect('articles.sqlite')
cur = conn.cursor()

cur.execute('''
DROP TABLE IF EXISTS Counts''')

cur.execute('''
CREATE TABLE Counts (author_unique TEXT , count INTEGER )''')

cur.execute('''
SELECT id, author FROM Articles ''')

all_rows = cur.fetchall()
# print(all_rows)

authors_list = [x[1] for x in all_rows]
# print "authors: ", authors_list

ids = [x[0] for x in all_rows]
#print ids

# author_lst = [item for sublist in all_rows[1] for item in sublist]

# Count articles per author
for id, author in itertools.izip(ids, authors_list) :
    authors = author.split('; ')
    for i in authors:
        parser = HTMLParser()
        parser = parser.unescape(i)

        cur.execute('SELECT count FROM Counts WHERE author_unique = ? ', (i, ))

        try:
            count = cur.fetchone()[0]
            # either increment the count by 1 if the author already exists in DB
            cur.execute('''UPDATE Counts SET count=count+1 WHERE author_unique = (?) ''', (i, ))

        except:
            # or insert a new row with a new author and count 1
            cur.execute('''
            INSERT INTO Counts VALUES (?, 1 )''', (i, ))
        conn.commit()


cur.close()

tt = time()-t0
print("Counting publications took: {}").format(round(tt,3))
