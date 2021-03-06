#coding=utf-8
class matchrow:
  def __init__(self,row,allnum=False):
    if allnum:
      self.data=[float(row[i]) for i in range(len(row)-1)]
    else:
      self.data=row[0:len(row)-1]
    self.match=int(row[len(row)-1])

def loadmatch(f,allnum=False):
  rows=[]
  for line in open(f):
    rows.append(matchrow(line.split(','),allnum))
  return rows
 
from pylab import *
def plotagematches(rows):
  xdm,ydm=[r.data[0] for r in rows if r.match==1],\
          [r.data[1] for r in rows if r.match==1]
  xdn,ydn=[r.data[0] for r in rows if r.match==0],\
          [r.data[1] for r in rows if r.match==0] 
  
  plot(xdm,ydm,'bo')
  plot(xdn,ydn,'b+')
  
  show()

def lineartrain(rows):
  averages={}
  counts={}
  
  for row in rows:
    # Get the class of this point
    cl=row.match
    
    averages.setdefault(cl,[0.0]*(len(row.data)))
    counts.setdefault(cl,0)
    
    # Add this point to the averages
    for i in range(len(row.data)):
      averages[cl][i]+=float(row.data[i])
      
    # Keep track of how many points in each class
    counts[cl]+=1
    
  # Divide sums by counts to get the averages
  for cl,avg in averages.items():
    for i in range(len(avg)):
      avg[i]/=counts[cl]
  
  return averages

def dotproduct(v1,v2):
  return sum([v1[i]*v2[i] for i in range(len(v1))])

def veclength(v):
  return sum([p**2 for p in v])

def dpclassify(point,avgs):
  b=(dotproduct(avgs[1],avgs[1])-dotproduct(avgs[0],avgs[0]))/2
  y=dotproduct(point,avgs[0])-dotproduct(point,avgs[1])+b
  if y>0: return 0
  else: return 1

def yesno(v):
  if v=='yes': return 1
  elif v=='no': return -1
  else: return 0
  
def matchcount(interest1,interest2):
  l1=interest1.split(':')
  l2=interest2.split(':')
  x=0
  for v in l1:
    if v in l2: x+=1
  return x

yahookey="dj0yJmk9eEhQSGhCN1Y0aGtQJmQ9WVdrOVpYTTBkRlEzTjJjbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD0xNA--"
from xml.dom.minidom import parseString
from urllib.request import urlopen
from urllib.parse import quote

loc_cache={}
def getlocation(address):
  if address in loc_cache: return loc_cache[address]
  data=urlopen('http://local.yahooapis.com/MapsService/V1/'+\
               'geocode?appid=%s&location=%s' %
               (yahookey,quote_plus(address))).read()
  doc=parseString(data)
  lat=doc.getElementsByTagName('Latitude')[0].firstChild.nodeValue
  long=doc.getElementsByTagName('Longitude')[0].firstChild.nodeValue  
  loc_cache[address]=(float(lat),float(long))
  return loc_cache[address]

def milesdistance(a1,a2):
  lat1,long1=getlocation(a1)
  lat2,long2=getlocation(a2)
  latdif=69.1*(lat2-lat1)
  longdif=53.0*(long2-long1)
  return (latdif**2+longdif**2)**.5

def stringdistance(str1,str2):
    length1=len(str1)
    length2=len(str2)
    return (length1**2+length2**2)**.5

def loadnumerical():
  oldrows=loadmatch('matchmaker.csv')
  newrows=[]
  for row in oldrows:
    d=row.data
    data=[float(d[0]),yesno(d[1]),yesno(d[2]),
          float(d[5]),yesno(d[6]),yesno(d[7]),
          matchcount(d[3],d[8]),
          stringdistance(d[4],d[9]),
          row.match]
    newrows.append(matchrow(data))
  return newrows

def scaledata(rows):
  low=[999999999.0]*len(rows[0].data)
  high=[-999999999.0]*len(rows[0].data)
  # Find the lowest and highest values
  for row in rows:
    d=row.data
    for i in range(len(d)):
      if d[i]<low[i]: low[i]=d[i]
      if d[i]>high[i]: high[i]=d[i]
  
  # Create a function that scales data
  def scaleinput(d):
     return [(d[i]-low[i])/(high[i]-low[i])
            for i in range(len(low))]
  
  # Scale all the data
  newrows=[matchrow(scaleinput(row.data)+[row.match])
           for row in rows]
  
  # Return the new data and the function
  return newrows,scaleinput


def rbf(v1,v2,gamma=10):
  dv=[v1[i]-v2[i] for i in range(len(v1))]
  l=veclength(dv)
  return math.e**(-gamma*l)

def nlclassify(point,rows,offset,gamma=10):
  sum0=0.0
  sum1=0.0
  count0=0
  count1=0
  
  for row in rows:
    if row.match==0:
      sum0+=rbf(point,row.data,gamma)
      count0+=1
    else:
      sum1+=rbf(point,row.data,gamma)
      count1+=1
  y=(1.0/count0)*sum0-(1.0/count1)*sum1+offset

  if y>0: return 0
  else: return 1

def getoffset(rows,gamma=10):
  l0=[]
  l1=[]
  for row in rows:
    if row.match==0: l0.append(row.data)
    else: l1.append(row.data)
  sum0=sum(sum([rbf(v1,v2,gamma) for v1 in l0]) for v2 in l0)
  sum1=sum(sum([rbf(v1,v2,gamma) for v1 in l1]) for v2 in l1)
  
  return (1.0/(len(l1)**2))*sum1-(1.0/(len(l0)**2))*sum0


agesonly=loadmatch('agesonly.csv',allnum=True)
#matchmaker=loadmatch('matchmaker.csv')
#plotagematches(agesonly)

#开始 单特征年龄线性分类
#lineartrain(agesonly)
#avgs=[[26.9,35.8],[35.4,33.1]]
#dpclassify([30,30],avgs)
#getlocation()
#结束 单特征年龄线性分类

#开始 多特征线性分类 数据缩放化
numericalset=loadnumerical()
scaledset,scalef=scaledata(numericalset)
#avgs=lineartrain(scaledset)
#print numericalset[0].data
#print numericalset[0].match
#print dpclassify(scalef(numericalset[0]),avgs)
#结束 多特征线性分类

#开始
#offset=getoffset(agesonly)
#print nlclassify([30,30],agesonly,offset)
#结束

#开始 测试svm sklearn
#from sklearn import svm
#from sklearn.svm import SVC
#import numpy as np
#数据集合
#X = np.array([
#  [1,0,1],
#  [-1, 0, -1]
#])
#y = np.array([1,-1])
#训练
#clf = SVC()
#clf.fit(X, y)
#预测
#y_t = clf.predict([1,-1,1])
#print(y_t )
#评分
#print(clf.score([1,-1,1],y_t))
#结束

#开始 对缩放化的转换后的数据 训练
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
answers,inputs = [r.match for r in scaledset],[r.data for r in scaledset]
#开始训练
clf = SVC()
print(clf)
#交叉验证
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, inputs, answers, cv=5)
print(scores)

clf.fit(inputs, answers)
#预测
y_t1 = clf.predict(np.array(scalef([28.0,-1,-1,26.0,-1,1,2,0.8])).reshape(1, -1))
print(y_t1)
y_t2 = clf.predict(np.array(scalef([28.0,-1,1,26.0,-1,1,2,0.8])).reshape(1, -1))
print(y_t2)

#结束


