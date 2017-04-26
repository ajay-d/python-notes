# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:37:28 2015

@author: adeonari
"""

import re

match_object = re.search('iig', 'piiig')
match_object

match_object.group()

#value None counts as false
def Find(pattern, text):
    match = re.search(pattern, text)
    if match: 
        print(match.group())
    else:
        print('not found')


Find('iig', 'piiig')
Find('abc', 'piiig')

# . any character except new line
Find('...g', 'piiig')
Find('c\.', 'c.all')

#raw
Find(r'c.', 'c.all')
Find(r'c\.', 'c.all')
Find('c.', 'c.all')

Find(r'c.', 'call')
Find(r'c\.', 'call')
Find('c.', 'call')


#\w word char (non special)
#\s whitespace
#\S non-whitespace
Find(r':\w\w\w', 'test :cat anything')
Find(r'\d\d\d', 'test :159 space123')
Find(r'\d\s\d\s\d', 'test :1 59 space1 2 3 ')

Find(r':\w+', 'test :cat999==&&++ anything')
Find(r':\S+', 'test :cat999==&&++ anything')

Find(r'\w+@\w+', 'text nick.d@gmail.com text')

#bracket to match set of characters
#Match the . (dot)
#no need to escape in the brackets
#once in the bracket, order doesn't matter
Find(r'[\w.]+@\w+', 'text nick.d@gmail.com text')
Find(r'[\w.]+@\w+', 'text ..n.ick.d@gmail.com text')

#parens are not matched, just put around the groups we care about
#%%
m = re.search(r'([\w.]+)@([.\w]+)', 'text nick.d@gmail.com text')
m.group()
#%%

#first set of grouping
m.group(1)
m.group(2)
re.compile(r'([\w.]+)@([.\w]+)').groups
#%%

m.span()
len(m.span())

m = re.search(r'([\w.]+)(@)([.\w]+)', 'text nick.d@gmail.com text')
#Count number of captured groups
re.compile(r'([\w.]+)(@)([.\w]+)').groups
regexp = r'([\w.]+)(@)([.\w]+)'
re.compile(regexp).groups

#doesn't stop at first match, grabs all
re.findall(r'[\w.]+@[.\w]+', 'text nick.d@gmail.com text foo@bar')
re.findall(r'([\w.]+)@([.\w]+)', 'text nick.d@gmail.com text foo@bar')
t = re.findall(r'([\w.]+)@([.\w]+)', 'text nick.d@gmail.com text foo@bar')
type(t)
type(t[0])
t[0]
dir(re)

#dot doesn't match new line
#can add optional argument
re.search(r'[\w.]+@.+', 'text nick.d@gmail.com text this\t\n and that').group()
re.search(r'[\w.]+@.+', 'text nick.d@gmail.com text this\t\n and that', re.DOTALL).group()
print('text nick.d@gmail.com text this\t more text\n and that')
print(r'text nick.d@gmail.com text this\t more text\n and that')

#%%
str = 'purple alice@google.com, blah monkey bob@abc.com blah dishwasher'
re.search(r'([\w\.-]+)@([\w\.-]+)', str).group()
re.search(r'([\w\.-]+)@([\w\.-]+)', str).group(1)
re.search(r'([\w\.-]+)@([\w\.-]+)', str).group(2)
#keep the user (\1) but have yo-yo-dyne.com as the host.
re.sub(r'([\w\.-]+)@([\w\.-]+)', r'\1@yo-yo-dyne.com', str)



#%%


















