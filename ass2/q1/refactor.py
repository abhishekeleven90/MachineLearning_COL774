import re
filename = 'test_orig.data'
filehandle=open(filename,'r')
text = filehandle.read()
text = text.replace('nonad.','1')
text = text.replace('ad.','-1')
filehandle.close()
newfilename = 'test.data'
newfilehandle=open(newfilename,'w')
newfilehandle.write(text)
newfilehandle.close()
