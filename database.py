# Build SQLite Database
import sqlite3
from bs4 import BeautifulSoup
import urllib
from urlparse import urljoin
import string
from nltk import SnowballStemmer
import itertools
from time import time

t0 = time()

conn = sqlite3.connect('articles.sqlite')
cur = conn.cursor()

cur.execute('''
DROP TABLE IF EXISTS Articles''')

cur.execute('''
CREATE TABLE Articles (id TEXT PRIMARY KEY )''')

cur.execute('''
ALTER TABLE Articles ADD COLUMN 'author' BLOB''')

cur.execute('''
ALTER TABLE Articles ADD COLUMN 'abstract' TEXT''')



# Unique article ID
def get_id(url):
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    article_id = soup.find("meta", attrs={'name': "citation_arxiv_id" } )
    id = article_id.get('content')

    return id

def authors_list(url):
    # Create list with author names
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    #print(soup.prettify())

    authors_lst = list()
    for author in soup.find_all("meta", attrs={'name': "citation_author" } ) :
        authors = author.get('content')
        #print "list: ", authors

        authors_lst.append(authors.encode('ascii', 'xmlcharrefreplace'))
    authors_lst = '; '.join(authors_lst)
    #print "authors: ", authors_lst
    return authors_lst

def abstract_text(url):
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    abstract_txt = soup.find('blockquote').text.lower()
    abstract_txt = abstract_txt[11:]
    return abstract_txt

def stem(text):
    stems = []
    exclude = set(string.punctuation)
    words = text.split()
    for word in words:
        # remove punctuation
        word = ''.join(ch for ch in word if ch not in exclude)

        # remove digits
        word = ''.join( [i for i in word if not i.isdigit()] )

        # stem words
        word = SnowballStemmer("english").stem(word)
        stems.append(word)
    stems = ' '.join(stems)

    return stems

# loop over Computer Science for different months
pages = ['0', '2000']
years = ['17', '16', '15', '14', '13', '12']
months = ['01','02', '03', '04','05','06', '07', '08', '09', '10', '11', '12']
for page in pages:
    for year in years:

        for month in months:

            search_url = urljoin('http://xxx.lanl.gov/list/cs/', str(year + month + '?skip=' + page + '&show=2000'))

            print(search_url)

            html = urllib.urlopen(search_url).read()
            soup = BeautifulSoup(html, "lxml")

            for article_id, i in itertools.izip(soup.find_all("a", attrs={'title': "Abstract" } ), range(len(soup.find_all("a", attrs={'title': "Abstract" } ))) ):
                #url = urljoin('https://arxiv.org', article_id.get("href"))
                url = urljoin('http://xxx.lanl.gov/', article_id.get("href"))
                article_id = get_id(url)
                authors = authors_list(url)

                abstract_txt = abstract_text(url)
                stems = stem(abstract_txt)

                params = (article_id, authors, stems)

                cur.execute('''
                INSERT OR IGNORE INTO Articles VALUES (?, ?, ?)''', params)
                conn.commit()

conn.close()

tt = time()-t0
print("Buidling DB took: {}").format(round(tt,3))
