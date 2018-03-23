#from urllib.request import urlretrieve
#from urllib.request import urlopen
#from bs4 import BeautifulSoup

#html = urlopen("http://www.pythonscraping.com")
#bsObj = BeautifulSoup(html)
#imageLocation = bsObj.find("a", {"id": "logo"}).find("img")["src"]
#urlretrieve (imageLocation, "logo.jpg")


#import os
#import re
#from urllib.request import urlretrieve
#from urllib.request import urlopen
#from bs4 import BeautifulSoup

#downloadDirectory = "downloaded"
#baseUrl = "http://pythonscraping.com"

#def getAbsoluteURL(baseUrl, source):
#    if source.startswith("http://www."):
#        url = "http://"+source[11:]
#    elif source.startswith("http://"):
#        url = source
#    elif source.startswith("www."):
#        url = source[4:]
#        url = "http://"+source
#    else:
#        url = baseUrl+"/"+source
#    if baseUrl not in url:
#        return None
#    return url

#def getDownloadPath(baseUrl, absoluteUrl, downloadDirectory):
#    path = absoluteUrl.replace("www.", "")
#    path = path.replace(baseUrl, "")
#    path = re.sub("\?.*", "", path)  # 把问号开头的字符串都替换为空
#    path = downloadDirectory+path
#    directory = os.path.dirname(path)
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    return path

#html = urlopen("http://www.pythonscraping.com")
#bsObj = BeautifulSoup(html)
#downloadList = bsObj.findAll(src=True)
#for download in downloadList:
#    fileUrl = getAbsoluteURL(baseUrl, download["src"])
#    if fileUrl is not None:
#        print(fileUrl)
#        try:
#            urlretrieve(fileUrl, getDownloadPath(baseUrl, fileUrl, downloadDirectory))
#        except urlopen.urllib.request.ContentTooShortError:
#            print
#            'Network conditions is not good.Reloading.'
#        #urlretrieve(fileUrl, getDownloadPath(baseUrl, fileUrl, downloadDirectory))

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random
import pymysql

conn = pymysql.connect(host='192.168.9.109', unix_socket='/tmp/mysql.sock',user='root', passwd='root', db='python', charset='utf8')
cur = conn.cursor()

def store(title, content):
    cur.execute("USE python")
    cur.execute("INSERT INTO pages (title, content) VALUES (\"%s\",\"%s\")", (title, content))
    random.seed(datetime.datetime.now())
    cur.connection.commit()

def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org"+articleUrl)
    bsObj = BeautifulSoup(html)
    title = bsObj.find("h1").get_text()
    content = bsObj.find("div", {"id":"mw-content-text"}).find("p").get_text()
    store(title, content)
    return bsObj.find("div", {"id":"bodyContent"}).findAll("a",href=re.compile("^(/wiki/)((?!:).)*$"))

links = getLinks("/wiki/Kevin_Bacon")
try:
    while len(links) > 0:
        newArticle = links[random.randint(0, len(links)-1)].attrs["href"]
        print(newArticle)
        links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()