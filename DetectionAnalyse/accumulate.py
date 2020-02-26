# Opening file 
file1 = open('HOGDetection.log', 'r')
#file1 = open('YOLODetection.log', 'r') 
count = 0
conCounter = 0
detection = 0.0
fps = 0.0 
# Using for loop 
print("Using for loop") 
for line in file1: 
    count += 1
    line =  line.strip()
    dLine = line
    if dLine[11:21] == 'detection:':
        #print(float(line[21:]))
        detection += float(dLine[21:])
        #print(detection)
    
    cLine = line
    #print(fLine[11:15])
    if cLine[11:23] == 'confidences:':
        conCounter += 1
    
    fLine = line
    #print(fLine[11:15])
    if fLine[11:15] == 'fps:':
        #print(float(line[21:]))
        fps += float(fLine[15:])
        #print(fps)
    
    
    
    #print("Line{}: {}".format(count, line)) 
    
print("detection avg = " + str((detection/90)))
print("fps avg = " + str((fps/90)))
print("confidences pro frame avg = " + str((conCounter/90)))

# Closing files 
file1.close() 