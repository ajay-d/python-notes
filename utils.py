import numpy as np
import pandas as pd
import pylab as pl
from datetime import time

import sys
import platform

import psutil

import os

help(os.listdir)

os.path.abspath("H:\Python")
os.path.exists("H:\Python")

os.listdir(os.path.abspath("H:\Python"))

from os import path, getcwd, chdir

def print_my_path():
    print('cwd:     {}'.format(getcwd()))
    print('__file__:{}'.format(__file__))
    print('abspath: {}'.format(path.abspath(__file__)))

print_my_path()

chdir('..')

print_my_path()

import shutil
#file copying utility
help(shutil.copy)

import subprocess
help(subprocess.getstatusoutput)
subprocess.getstatusoutput('dir')
subprocess.call(['dir'], shell=True)

#Python 3
print(sys.version)
print('Python', platform.python_version())

sys.maxsize
sys.path
sys.path.count('3')

dir(sys)
help(sys)
help(sys.exit)

len
help(len)

from pathlib import Path
Path.cwd()

pd.__version__
np.__version__

psutil.phymem_usage()
psutil.virtual_memory()

pd.show_versions(as_json=False)

os.getcwd()
os.path.dirname(os.path.realpath(__file__))

####
#types
type('string')
isinstance("test string", str)

type(True)
type(None)

a = 1.0
print(type(a))

print(3/4)
print(3.0 / 4.0)

for i in range(1,10):
    print(i)

a = 23
if a >= 22:
   print("if")
   print("greater than or equal 22")
elif a >= 21:
    print("elif")
else:
    print("else")
    
a = "1"
try:
  b = a + 2
except:
  print(a, " is not a number")
  
def Division(a, b):
    print(a/b)
Division(3,4)
#Division(3,"4")

def Division(a, b):
    try:
        print(a/b)
    except:
        if b == 0:
           print("cannot divide by zero")
        else:
           print(float(a)/float(b))
Division(2,"2")
Division(2,0)

a = "A string of characters, with newline \n CAPITALS, etc."
print(a)
b=5.0
newstring = a + "\n We can format strings for printing %.2f"
print(newstring %b)

print(a.find('s'))
a.count('c')
range(100)
even_numbers = [x for x in range(100) if x % 2 == 0]
print(even_numbers)


def sqr(x): 
    return x ** 2
a = [2,3,4]
b = [10,5,3]
c = map(sqr,a)
print(c)
print(list(c))
d = map(pow,a,b)
list(d)

'hello'.upper()

sys.getsizeof(a)
sys.getsizeof(pl.rand(1000))

# In many programming languages, loops use an index.
# This is possible in Python, but it is more
# idiomatic to use the enumerate function.
# using and index in a loop
xs = [1,2,3,4]
for i in range(len(xs)):
    print (i, xs[i])

# using enumerate
for i, x in enumerate(xs):
    print (i, x)

###
#import custom code / modules
print(sys.path)
sys.path.append(r'H:\Python\google-python-exercises\babynames\solution')
print(sys.path)

import babynames
dir(babynames)

babynames.extract_names(r'H:\Python\google-python-exercises\babynames\baby1990.html')

import sys
print(sys.path)
sys.path.append(r'H:\Python')
print(sys.path)

import person_class
#import imp
import importlib

#http://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide
dir(person_class)

p1 = person_class.Person(12)
p1.get_population()

p2 = person_class.Person(63)
p1.get_population()

p1.get_age()
p2.get_age()

#or directly
p1.age
p1.population

#first argument to function is self
#These are exactly equivalent calls
p1.get_age()
person_class.Person.get_age(p1)
person_class.Person.get_age(p2)

#Class or static variable
person_class.Person.population

#Reload a previously imported module
#imp.reload(person_class)
importlib.reload(person_class)

p1.clear()
person_class.Person.population

#can set the class attribute and this affects all instances
person_class.Person.population = 10
p1.get_population()
p2.get_population()

c = person_class.Circle(10)
c.pi
c.area()

joe = person_class.Person2('Joe')
bob = person_class.Person2('Bob')
person_class.Person2.all_names

#Fetch Internet Resources Using The urllib Package
#https://docs.python.org/3/howto/urllib2.html
import urllib

uf_local, uf_headers = urllib.request.urlretrieve("http://www.nber.org/cycles.html")
html = open(uf_local)
type(html)

uf = urllib.request.urlopen("http://www.nber.org/cycles.html")
uf.read()
uf.geturl()
uf.info()

uf = urllib.request.urlopen("http://www.nber.org/cycles.html")
uf.read(100)
uf.read().decode(uf.headers.get_content_charset())
text = uf.read().decode()

uf = urllib.request.urlopen("http://www.nber.org/cycles.html")
b = uf.read()
b.decode('utf-8')

nber = b.decode('utf-8')
re.search(r'Peak', nber)
re.search(r'Peak', nber).group()


f = urllib.request.urlopen('http://www.python.org/')
f.read()
print(f.read(100).decode('utf-8'))
type(f.read(100).decode('utf-8'))
f.read().decode(f.headers.get_content_charset())

















