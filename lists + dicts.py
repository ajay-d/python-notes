
import sys
#fucntion is tied to namespace
print(sys.argv)

from sys import argv
#now available by short name
print(argv)

dir(sys)
dir(list)
help(list.append)

#dictionary

#empty
d = {}
d['a'] = 'alpha'
d['b'] = 'beta'
d['o'] = 'omega'

len(d)
d
d2 = {'a': 'alpha', 'o': 'omega', 'g': 'gamma'}

d.get('a')
d.get('z')
'a' in d
'z' in d

None == d.get('z')

#random order
d.keys()

for i in d.keys():
    print(i)

d.values()

for k in sorted(d.keys()): 
    print('key:', d[k])

d.items()

for k, v in d.items(): 
    print (k, '>', v)

#lists
l1 = [1, 3, 5, 9, 3, 2]
sorted(l1)

#list comprehension
nums = [1, 2, 3, 4]
squares = [ n * n for n in nums ]
squares

l1_text = [str(x) for x in l1]
l1_text
''.join(l1_text)

l1.append(9)
l1

l2 = ['a', 'b', 'z']
[len(a) for a in l2]
':'.join(l2)
l2.pop()

s = ':'.join(l2)
s
s.split(':')

'aaa bbb ccc'.split()

range(10)

l3 = []
for i in range(10):
    l3.append(i)
l3

len(l3)

#list of tuples
t = [(1,'b'), (2,'a'), (1,'a')]
sorted(t)

raw = r'this\t\n and that'
raw
print(raw)

##List comprehension
import math

points = [(2,3), (4,8), (5,10)]
for a,b in points:
    print(a)

diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
diffs_squared_distance
sum(diffs_squared_distance)
math.sqrt(sum(diffs_squared_distance))

##Slicing
df =[11.9, 14.2, 15.2, 16.4, 17.2, 18.1, 18.5, 19.4, 22.1, 22.6, 23.4, 25.1]
type(df)

#data[1:] is equivalent to data[1: len(data)]
#data[:-1] is equivalent to data[0: len(data) - 1]
df[1:]
df[:-1]
len(df)

##collections.Counter
import collections
symbols = ['o', 'x', 'o', 'o', 'x', '-', '-', '-']

collections.Counter(symbols)
count = collections.Counter(symbols)
count.most_common(2)
