filename = 'digitlabels.txt'
newfile  = 'digitlabels2.txt'
filename = open(filename,'r')
all = filename.readlines()
towrite = ''
for line in  all[1:]:
    line = line.split()
    towrite = towrite  + line[1] + '\n'
filename.close()
newfile= file(newfile,'w')
newfile.write(towrite)
newfile.close()

filename = 'digitdata.txt'
newfile  = 'digitdata2.txt'
filename = open(filename,'r')
all = filename.readlines()
towrite = ''
for line in  all[1:]:
    line = line.split()[1:]
    str1 = ' '.join(line)
    towrite = towrite  + str1 + '\n'
filename.close()
newfile= file(newfile,'w')
newfile.write(towrite)
newfile.close()
