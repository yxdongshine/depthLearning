#coding=utf-8
"""
集体智慧样例代码地址：
    http://shop.oreilly.com/product/9780596529321.do
源代码git:https://resources.oreilly.com/examples/9780596529321/

python pip 安装教程：https://jingyan.baidu.com/article/e73e26c0d94e0524adb6a7ff.html
        因为是用的anaconda3的Python3处理的，切换到anaconda3的目录下D:\yxd\tool\anaconda\anaconda3\path\Scripts
        easy_install.exe pip
        pip install feedparser

feedparser官网：安装教程
https://pypi.python.org/pypi/feedparser
解决问题：https://www.cnblogs.com/space-place/p/6260521.html
"""

import feedparser
import re

#返回一个RSS订阅源的标题和包含的单词计数情况字典
def getwordcount(url):
    d = feedparser.parse(url)
    wc = {}
    #循环遍历所有的文章的条目
    for e in d: