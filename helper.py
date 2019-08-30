#!/usr/bin/env python
# coding: utf-8

find = ';'
replace = ','
source_filename = 'household_power_consumption.txt'
destination_filename = 'household_power_consumption.csv'

# Read in the file
with open(source_filename, 'r') as file :
    filedata = file.read()

# Replace the target string
filedata = filedata.replace(find, replace)

# Write the file out again
with open(destination_filename, 'w') as file:
    file.write(filedata)
    
print('>> Successfully replaced %s with %s' % (find, replace))
