#coding=utf-8
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#匹配类
class matchrow:
  def __init__(self,row,allnum=False):
    if allnum:
      self.data=[float(row[i]) for i in range(len(row)-1)]
    else:
      self.data=row[0:len(row)-1]
    self.match=int(row[len(row)-1])

#加载数据
def loadmatch(f,allnum=False):
  rows=[]
  for line in open(f):
    rows.append(matchrow(line.split(','),allnum))
  return rows

#是否要小孩子
def yesno(v):
  if v=='yes': return 1
  elif v=='no': return -1
  else: return 0

#共同爱好的交集数
def matchcount(interest1,interest2):
  l1=interest1.split(':')
  l2=interest2.split(':')
  x=0
  for v in l1:
    if v in l2: x+=1
  return x

#住址的远近距离
def stringdistance(str1,str2):
    length1=len(str1)
    length2=len(str2)
    return (length1**2+length2**2)**.5

#加载数据并且转换成数值类型
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



#思路：
#    1.加载数据
#      非数值类型特征通过某种距离转化成数值类型（共同爱好的交集数，住址的远近距离，是否要小孩子）
#    2.将数据转换成符合训练的格式的数组
#    3.将数据最大最小值标准化
#    4.
#      4.1利用GridSearchCV选取超参数
#      用于选择哪种分类或者核函数模型，特征等优化措施
#       好的参数等功能
#      4.2交叉训练验证
#
#    5.将数据拆分成训练集和测试集
#    6.预测值
#       将数据随机拆分一部分数据出来预测



#开始 多特征线性分类
numericalset = loadnumerical()
#转换成符合训练的数据格式
Y,X = [r.match for r in numericalset],[r.data for r in numericalset]
#Y,X = np.array(Y,dtype=np.float),np.array(X,dtype=np.float)
#数据最大最小值标准化
min_max_scaler = preprocessing.MinMaxScaler()
#其中目标值Y不需要标准化 因为已经是0或者1
X_stand = min_max_scaler.fit_transform(X)
#4.1 GridSearchCV
#超参数
tuned_parameters = [{'kernel': ['rbf'] ,'gamma': [1e-3, 1e-4]},
                    {'kernel': ['linear'],'degree':[3,5,7,9]}]
clf = SVC()
C_s = np.logspace(1, 10, 100, 1000)
scores = list()
scores_std = list()
for C in C_s:
    #交叉验证
    clf.C = C
    this_scores = cross_val_score(clf, X_stand, Y, cv=5)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

    #GridSearchCV选取超参数
    clfg = GridSearchCV(SVC(), tuned_parameters, cv=5)
    clfg.c = C
    clfg.fit(X_stand, Y)
    print("Best parameters set found on development set:")
    print()
    print(clfg.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clfg.cv_results_['mean_test_score']
    stds = clfg.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clfg.cv_results_['params']):
        print("%0.9f (+/-%0.09f) for %r"
              % (mean, std * 2, params))
    print()

#画图
#plt.figure(1, figsize=(8, 6))
#plt.clf()
#plt.semilogx(C_s, scores)
#plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
#plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
#locs, labels = plt.yticks()
#plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
#plt.ylabel('CV score')
#plt.xlabel('Parameter C')
#plt.show()

#开始预测 根据上面得到的最优的参数 和原始默认值比较
X_train, X_test, y_train, y_test = train_test_split(X_stand, Y, test_size=0.4, random_state=0)
#线性参数模型
clf_line = SVC(kernel='linear')#定义模型
clf_line.fit(X_train,y_train)#训练
y_pre_line = clf_line.predict(X_test)#预测y值集合
clf_line_score=clf_line.score(X_test, y_test)#计算得分
print(clf_line_score)
#最优化参数模型
clf_rbf = SVC(kernel='rbf',C=100)
clf_rbf.fit(np.array(X_train,dtype=np.float),np.array(y_train,dtype=np.float))
y_pre_rbf = clf_rbf.predict(X_test)
clf_rbf_score=clf_rbf.score(X_test, y_test)#计算得分
print(clf_rbf_score)

#画图展示预测真实值和线性，rbf
plt.figure(2, figsize=(80, 6))
plt.clf()
#plt.plot(X_test[:,0], y_test,  color='black') 连续线图
plt.scatter(X_test[:,0], y_test,  color='black',linewidth=3) #点图
#plt.scatter(X_test[:,0], y_pre_line, color='blue', linewidth=3)
plt.scatter(X_test[:,0], y_pre_rbf, color='red', linewidth=3)
plt.xticks((X_test[:,0]))
plt.yticks((y_test))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
