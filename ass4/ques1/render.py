filename = open('digitdata.txt','r')
first = filename.readlines()[0]
indexlist = []
for item in first.split():
    val = item.split('"')[1].split('pixel')[1]
    indexlist.append(int(val))
filename.close()
#print indexlist
#print len(indexlist)

print 'Enter image index'
indextoprocess = int(raw_input())
indextoprocess = indextoprocess - 1

filename = open('digitdata.txt','r')
all = filename.readlines()[1:]
curr = all[indextoprocess]
filename.close()

tempdict = {}
split = curr.split()
dataline = split[1:]


for i in range(len(dataline)):
    tempdict[indexlist[i]] = int(dataline[i])
line = ''

for i in range(784):
    if i in tempdict:
        line = line + str(tempdict[i])+ ' '
    else:
        
        line = line + str('0'+' ')
#print len(line.split())
line = line.split()

import numpy
img = numpy.zeros((28,28))
#print img

vectorindex = 0

for i in range(28):
  for j in range(28):
      img[i][j] = line[vectorindex];
      vectorindex = vectorindex + 1;

from pylab import *
import matplotlib.pyplot as plt
plt.imshow(img, interpolation='none',cmap=gray())
plt.show()
