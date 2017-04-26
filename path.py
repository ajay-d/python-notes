# -*- coding: utf-8 -*-
import sys
import os

def main():
  # Get the name from the command line, using 'World' as a fallback.
  if len(sys.argv) >= 2:
    name = sys.argv[1]
  else:
    name = 'World'
  print ('Hello', name)
  print ('os.getcwd()= ', os.getcwd())
  print('real path', os.path.realpath(name))
  print('dirname', os.path.abspath(os.path.dirname(name)))

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
  main()