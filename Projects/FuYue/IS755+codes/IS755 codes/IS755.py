import sys
import os.path
import re
import shelve

veShelve = shelve.open("venueShelve")
veShelve.clear()
file=open("checking_local_dedup.txt","r")
for line in file:
        tokens=line.split(',')
	venueid=tokens[1]
	veShelve[venueid]=1	

file.close()


# 48969 unique venues

cnt = 0
file=open("venue_info.txt","r")
for line in file:
        tokens=line.split(',')
        venueid=tokens[0]
	if veShelve.has_key(venueid):
		cnt = cnt+1;

file.close()

print str(cnt)

veShelve.close()

