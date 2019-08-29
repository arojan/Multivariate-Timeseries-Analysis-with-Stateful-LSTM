#!/usr/bin/env python
# coding: utf-8

find = ';'
replace = ','

# Read in the file
with open('household_power_consumption.csv', 'r') as file :
    filedata = file.read()

# Replace the target string
filedata = filedata.replace(find, replace)

# Write the file out again
with open('household_power_consumption.csv', 'w') as file:
    file.write(filedata)
    
print('>> Successfully replaced %s with %s' % (find, replace))