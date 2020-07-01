import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
"""headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '3600',
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
    }"""
url = "https://en.wikipedia.org/wiki/Google"
req = requests.get(url)
soup = BeautifulSoup(req.content, 'html.parser')
print(type(soup))
print(type(soup.prettify()))
# get text
text = soup.body.get_text()
#print(text)

# break into lines and remove leading and trailing space on each
lines = [line.strip() for line in text.splitlines()]
print(lines)
# break multi-headlines into a line each
chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]
print(chunks)
# drop blank lines
text = ' '.join(chunk for chunk in chunks if chunk)
print(text)
print(type(text))

# Saving to a Text File
with open('Input', 'w') as text_file:
    text_file.write(str(text.encode("utf-8")))
"""my_data_file = open('input.txt', 'w', encoding='utf-8')
my_data_file.write(str(soup))
my_data_file.close()
"""
f = open('input','r',encoding='utf-8')
read_data1 = f.read()
print(type(read_data1))
stokens = sent_tokenize(read_data1)
#for s in stokens:
print(stokens)
wtokens = word_tokenize(read_data1)
for w in wtokens:
    print(w)


