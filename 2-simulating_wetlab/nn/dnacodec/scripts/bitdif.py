import sys
try:
    with open(sys.argv[1],'rb') as file1:
        contentsofF1= file1.read()
    
    with open(sys.argv[2],'rb') as file2:
        contentsofF2= file2.read()
#Raise Exception if unable to open files
except Exception:
    print("Unable to open file(s)")
    exit()

#Exiting if the sizes of the 2 files are different
if(len(contentsofF1)!=len(contentsofF2)):
    print("Length of the file is not equal")
    exit()
#Looping over the chars
for x in range(len(contentsofF1)):
    #Checking bits of char only when different
    if contentsofF1[x]!=contentsofF2[x]:
        #Looping over the bits
        for y in range(8):
            #Masking logic
            mask=((0x80)>>y)
            maskedF1= contentsofF1[x] & mask
            maskedF2= contentsofF2[x] & mask
            #Printing bit number if different
            if(maskedF1 != maskedF2):
                print(x*8 + y)
