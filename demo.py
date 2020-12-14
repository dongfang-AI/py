# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 10:10:35 2020

@author: wh
"""
#第一个程序
#a = 1
#b = 2
#sum = a + b
#print("sum is:{}".format(sum))
#print("sum is:{}".format(a+b))


#import numpy as np
#
#a = np.zeros((2,3))
#print("2行3列的0矩阵：".format(a))
#[m,n]=a.shape
#print("0矩阵的行m为{}，列n为{}".format(m,n))
#
#for i in range(0,m):
#    print(i)

#集合
#name = {'Sarry','Tom','Kindy'}  
#if 'Rose' in name:
#    print("Rose is the student!")
#else:
#    print("Rose is not the studnet!")

#集合运算
#a = set('abasd')
#b = set('abassss')
#print(a,b)
#print(a-b)
#print(a|b)
#print(a&b)
#print(a^b)

#字典
#tinydict={'name':'runoob','code':'1','site':'www.runoob.com'}
#print(tinydict)
#print(tinydict.keys())
#print(tinydict.values())

#输入函数 input()
#dec = int(input("输入数字："))
#print("十进制：",dec)
#print("二进制：",bin(dec))
#print("八进制：",oct(dec))
#print("十六进制：",hex(dec))

#if-else语句
#age = input("你多大了？")
#if int(age) > 18: 
#    print("你已经是个大人啦！干杯！")
#else: 
#    print("你还没有成年，不能饮酒！")

#if-elif 语句
#grades= float(input("你的期末成绩是多少？"))
#if grades > 90:
#    print("成绩：优秀！")
#elif grades > 80:
#    print("成绩：良好！")
#elif grades > 60:
#    print("成绩：及格！")

#while循环语句
#times = 10000
#while times > 0:
#    print("媳妇，我错了！")
#    times -= 1
#    print("第%d次\n"%(times))

#星谱
#i=1
#while i <=5: #i行
#    j=1
#    while j<=i: #每行i个
#        print("* ",end='')
#        j+=1
#    print('\n')
#    i+=1

#99乘法表
#i = 1
#while i<=9:
#    j=1
#    while j <= i:
#        print("%d*%d=%-2d "%(j,i,i*j),end='')
#        j+=1
#    print('\n')
#    i+=1

#str1 = input("祝福语？")
#for s in str1:
#    if s == '中':
#        break
#    if s == "。":
#        continue
#    print("{}".format(s))


#i=-1
#while i >=-3:
#    print(str2[i],end="")
#    i-=1

#列表的循环打印
#L1=['张伟','王冲','我进山']
#for s in L1:
#    print(s)

#列表中增加元素
#L1=['张伟','王冲','吴金山']
#L2=['张敦','李茂']
#L1.append('吴金山')
#L1.extend(L2)
#L1.insert(6,'晚饭')
#L1[-1]='王峰'
#
#length = len(L1)
#i=0
#while i < length:
#    print(L1[i])
#    i+=1
#    
#name = input("账号名称：")
#if name in L1:
#    print("登陆成功！")
#elif name not in L1:
#    print("非法用户！")
#
#print("你的ID是：%d"%(L1.index(name)))
#print("重名用户有%d个"%(L1.count(name)))
#
#del L1[2] #删除指定下标元素
#L1.pop() #删除最后一个元素
#L1.remove('李茂') #删除指定元素
#print(L1)

#列表排序
#a = ['a','c','d','b']
#b=['4','1','3','2']
#
#a.reverse() #反序
#b.sort(reverse=True) #降序
#print(a)
#print(b)

#列表嵌套
#nameList = [['Tokee','Liny','Tom'],['tengxun','baidu','xiaomi'],['李畅','李顺','王骞']]
#print("Class1 名单："+str(nameList[0]))
#print("Class2 名单："+str(nameList[2][0]))

#元组
#grades = ('语文',12,90.9)
#print(grades[0],grades[2])

#字典
info = {'name':'班长','id':100,'sex':'f','address':'湖北武汉'}
#元素提取
#id = info.get('age','001') #提取age对应值，默认001
#print(info['name'],id)
#print(str(info['name'])+str(id))
#修改元素
#newId=input("新用户ID：")
#print("初始ID：",info['id'])
#info['id']=newId
#print("更新ID：",info['id'])

#添加新元素
#info['Tel']=input("电话：")
#print(info)

#删除元素
#del info['sex'] #删除字典中sex:f键值对
#del info #删除字典
#info.clear() #情况字典info
#print(info)

#字典遍历
#print(len(info)) #info字典的长度
#for key in info.keys(): #info的键
#    print(key)
#for value in info.values(): #info的值
#    print(value)
#for item in info.items(): #info的键值对
#    print(item)
#for key,value in info.items():
#    print(key+':'+value)

##字符串遍历
#str1 = 'hello, 123456'
#print('字符串长度：',len(str1))
#i=1
#for s in str1:
#    print(i,':',s)
#    i+=1

##列表遍历
#L1 = ['王菲',35,'歌手',1.65,'中国国籍']
#i=0
#for l in L1:
#    print('{}:'.format(i),l,'')
#    i+=1

#i=1
#for str in L1:
#    print("%d:%s"%(i,str))
#    i+=1

#元组遍历
#a_t = (1,2,3,4,5,6)
#for a in a_t:
#    print(a)
#i=0
#while i < len(a_t):
#    print(a_t[i])
#    i+=1

#列表合并
#str1=['王菲',25]
#str2=['女',1.65]
#print(str1+str2)
#print(str1*2)

##元组合并
#t1=('王菲',26)
#t2=('女',1.63)
#print(t1+t2)
#print(t1*3)

#字符串合并
#s1='我是中国人，'
#s2='我热爱自己的祖国！'
#print(s1+s2)
#print(s1*2+s2)

#函数的定义
#def print_c(carrier='程序员'):
#    print('我是一名{}'.format(carrier))
#
#print_c()
#print_c('高级程序员')

#函数的调用与交互
#def add(num1=0,num2=0):
#    print("{}+{}的和为：{}".format(num1,num2,float(num1)+float(num2)))
#
#add()
#a,b=input("输入2个数：").split(',')
#add(a,b)

#函数传参：
#def register(name='Tom',age=24,gender='男'):
#    print("名字：%s\n年龄：%d\n性别：%s"%(name,int(age),gender))
#a,b,c=input("输入名字、年龄和性别，空格隔开！\n").split()
#register(a,b,c)
#register(gender=c,name=a)

#函数的返回值
#def secret(i):
#    if i == '1':
#        return '你是我的英雄！'
#    elif i == '2':
#        return '保家卫国！'
#    else:
#        return '前进！'
#
#name,i=input("输入你的名字和文件编号（1~2）").split()
#s1=secret(i)
#print("%s,%s"%(name,s1))

#函数嵌套
#def p_star(i):
#    print('* '*i)
#
#def num_star(j):
#    i=1
#    while i <= j:
#        p_star(i)
#        i+=1
#    while j > 0:
#        j-=1
#        p_star(j)
#
#num_star(int(input("星星阶数：")))

#局部变量和全局变量
#tea='龙井' #字符串为不可变数据类型
#def Tea(mingzi,cha):
#    global tea #声明全局变量，可修改
#    tea=cha
#    name=mingzi
#    print("%s,请喝%s!"%(name,tea))
#
#s1,s2=input("您是？喝什么茶？").split()
#Tea(s1,s2)
#print(tea)

#不定长参数
#def fun(a,b,*args,**kwargs):
#    print("a=",a)
#    print("b=",b)
#    print("args=",args) #元组
#    print("kwargs=",kwargs) #字典
#
#fun(1,2,3,'c','d',3,m=1,n=2)

#类的定义
#class Car:
#    def __init__(self): #构造函数，创建对象时，自动调用
#        self.wheelNum=4
#        self.color='黑色'
#        
#    def move(self):
#        print('车辆在奔驰！')
#    
#    def toot(self):
#        print('车辆在鸣笛...嘟嘟嘟')

#bmw=Car()
#bmw.move()
#bmw.color='黑色'
#print(bmw.color,bmw.wheelNum)



#构造函数
#class Car: 
#    def __init__(self,Num=4,color='黑色'):
#        self.wheelNum = Num
#        self.color=color
#        
#    def move(self):
#        print('车辆在奔驰！')
#    
#    def toot(self):
#        print('车辆在鸣笛...嘟嘟嘟')
#
#moto=Car(6,'红色')
#print(moto.wheelNum,moto.color)

#魔法方法
#class Car: 
#    def __init__(self,Num=4,color='黑色'):
#        self.wheelNum = Num
#        self.color=color
#    def __str__(self):
#        msg = '打印对象时调用'+self.color
#        return msg
#    
#    def move(self):
#        print('车辆在奔驰！')
#    
#    def toot(self):
#        print('车辆在鸣笛...嘟嘟嘟')
#
#moto=Car(6,'红色')
#print(moto)

#self
#class Animal:
#    def __init__(self,name):
#        self.name=name
#    
#    def printName(self):
#        print('名字为：{}'.format(self.name))
#    
#def myPrint(animal):
#    animal.printName()
#
#dog1 = Animal('小贝')
#myPrint(dog1)
#dog2 = Animal('阿黄')
#myPrint(dog2)

#保护私有属性
#class People:
#    def __init__(self,name):
#        self.__name=name #私有数据
#        
#    def getName(self):
#        return self.__name
#    
#    def setName(self,newName):
#        if len(newName)>5:
#            self.__name=newName
#        else:
#            print('名字长度需要至少5位！')
#    def __do(self): #私有方法
#        print('007工作制')
#    
#    def work(self):
#        self.__do() #调用私有方法
#        
#w=People('dongGe')
#print(w.getName())
#w.setName('dongdong2')
#print(w.getName())
#w.work()

#析构函数
#class Animal:
#    def __init__(self,long=5):
#        self.long=long
#        print('构造函数出动！')
#    
#    def run(self):
#        print('奔跑{}公里！'.format(self.long))
#    
#    def __del__(self):
#        print('析构函数收尾！')
#
#cat = Animal(25)
#cat.run()
#del cat
#dog = Animal()
#dog1=dog #对象引用计数为2
#del dog1
#print('删除dog1')
#del dog
#print('删除dog')

##继承关系
#class Animal:
#    def run(self):
#        print('调用Animal的函数！')
#    
#    def live(self):
#        print('保护野生动物！保护地球！')
#
#class Cat:
#    def __init__(self,name='大脸猫',color='黑色'):
#        self.name=name
#        self.color=color
#        print('Cat的构造函数被调用！')
#    
#    def run(self):
#        print('%s,自由的奔跑在沙发上！'%(self.name))
#
#class Tiger(Cat,Animal):
#    def __init__(self,name='东北虎',color='黄色'):
#        super().__init__(name,color)
#        
#    def setName(self,newName):
#        self.name=newName
#    
#    def eat(self,food):
#        print('{},正在吃{}'.format(self.name,food))
#    
##    def run(self):
##        print('{},扭到了腰！'.format(self.name))
#
#lucy = Tiger('东北虎1','黄色')
#print(lucy.name+' is '+lucy.color)
#lucy.run()
#lucy.eat('野猪腿')

#类属性
#class Student:
#    num = 0
#    def __init__(self,name='小城',age=20):
#        self.name=name
#        self.age=age
#        print('学生：'+self.name,self.age)
#        Student.num+=1
#
#stu1=Student()
#print('类属性：{}'.format(stu1.num))
#stu2=Student('小惠',18)
#stu2.num=10
#print('对象属性：%d'%stu2.num)
#del stu2.num
#print('类属性：{}'.format(stu2.num))

#类方法
#class Teacher:
#    num=20
#    def __init__(self,newNum):
#        self.num=newNum
#        
#    @classmethod #类方法
#    def getNum(cls):
#        return cls.num
#    @classmethod #类方法
#    def setNum(cls,newNum):
#        cls.num=newNum
#        return cls.num
#    @staticmethod #静态方法，无参数
#    def hello():
#        print('静态方法启动啦！')
#
#t1=Teacher(3)
#print('类属性num：%d,对象属性num：%d'%(Teacher.num,t1.num))
##类属性与对象属性同名，对象属性级别更高
#Teacher.num=1
#print('类属性num：%d'%Teacher.setNum(5))
#Teacher.hello()

#format()函数
#print('{} is a {}, her age is {}'.format('王菲','歌手',24))
#print('语文：{0} 数学：{2} 英语：{1}'.format(98,95,89))
#print('兴趣爱好：{0[0]}、{0[1]}、{0[2]}'.format(['写作','书法','排球']))
#print('菜单：{甜点}、{主食}、{水果}'.format(甜点='米酒汤圆',主食='刀削面',水果='香蕉'))
#print('Pi:{:.3f}'.format(3.1415926))
#print('{:o}'.format(17)) #进制b,o,d,x
#print('{:^10}'.format('abc')) #^居中，<左对齐，<右对齐
#print('%s,%d,%.3f'%('王菲',24,1.68))

#打开或创建test.txt文件
#f = open('test.txt','r+') 
#'r'只读，'w'只写，'r+'读写
#写入数据
#for i in [1,2,3]:
#    f.write('{}:hello,world\n'.format(i))
#读数据
#content=f.read(3) #读3个字符
#print(content)
#content=f.readlines() #列表格式
#for line in content:
#    print('{}'.format(line))
#
#f.close() #关闭文件

#文件拷贝
#oldFileName=input("输入文件名：")
#oldFile=open(oldFileName,'r')
#
#if oldFile:
#    fileFlagNum=oldFileName.rfind('.')
#
#    if fileFlagNum > 0:
#        fileFlag=oldFileName[fileFlagNum:]
#    
#    newFileName=oldFileName[:fileFlagNum]+'_copy_'+fileFlag
#    
#    newFile=open(newFileName,'w')
#    
#    for line in oldFile.readlines():
#        newFile.write(line)
#
#oldFile.close()
#newFile.close()

##异常处理
#try:
#    print('--test1--')
#    open('123.txt','r')
#    print('--test2--')
#except IOError:#处理IO异常
#    pass
#finally:
#    print('Done')
##异常处理
#try:
#    num=input("年龄？")
#    print(num)
#except Exception as e: #处理NameError
#    print('Error:{}'.format(e))
##    pass
#else:
#    print('Right')

#import math
#import numpy as np
#from test import add
#
#print(math.sqrt(10))
#print(np.array([1,2,3]))
#print(test.add(1,2))

#列表推导式
#a=[x for x in range(0,5)]
#b=[x for x in range(0,10) if x%2==0]
#c=[(x,y) for x in range(0,3) for y in range(0,2)]
#print(a)
#print(b)
#print(c)

import numpy as np

a = np.array([['王伟',20,'计算机'],['张浩',19,'法律']])
type(a)








