
import cv2
import numpy as np
dimensionFrameRecord = 20
minGoodPoint_A, maxFrame_AAdd, minFrame_ASet = 150, 20, 10
minGoodPoint_1, maxFrame_1Add, minFrame_1Set =  20, 20, 10

sift = cv2.SIFT_create()
#image1C = cv2.imread("./image/Skittles.jpg")  
image1C = cv2.imread("./imgScrew7.png")
image1G = cv2.cvtColor(image1C, cv2.IMREAD_GRAYSCALE)  
kp1, des1 = sift.detectAndCompute(image1G, None)

cap = cv2.VideoCapture("./IMG_8669.mov")
# cap = cv2.VideoCapture("./video_B5.avi")
ret, frame = cap.read()
height, width = frame.shape[:2]
posTextObj1 = (30, int(0.10*height))    # 10%
posTextObjA = (30, int(0.20*height))    # 20%
runGoodArray1 = np.zeros(dimensionFrameRecord)
runGoodArrayA = np.zeros(dimensionFrameRecord)
Obj1_Total, Obj1_Adder = 0, 0
ObjA_Total, ObjA_Adder = 0, 0



scale_percent = 40
width = int(int(cap.get(3)) * scale_percent/100)
height = int(int(cap.get(4)) * scale_percent/100)
dim = (width,height)


while(cap.isOpened()):
    ret, image2C = cap.read()
    resized = cv2.resize(image2C,dim,interpolation=cv2.INTER_AREA)
    if ret == True:
        for iShift in range(dimensionFrameRecord-1):
            runGoodArray1[iShift] = runGoodArray1[iShift+1]
            runGoodArrayA[iShift] = runGoodArrayA[iShift+1]
            
        image2G = cv2.cvtColor(resized, cv2.IMREAD_GRAYSCALE) 
        kp2, des2 = sift.detectAndCompute(image2G, None)
        #-----------------------------------------------------------------------
        good = []
        if len(kp2) != 0 :
            match = cv2.BFMatcher()
            matches = match.knnMatch(des1, des2, k=2)
            for i_matche in range(len(matches)):
                try:
                    m, n = matches[i_matche]
                except (ValueError):
                    pass
                else:
                    if m.distance < 0.5 * n.distance :
                        good.append(m)
        
        if len(good) >= minGoodPoint_1:
            runGoodArray1[dimensionFrameRecord-1] = 1  
        else:
            runGoodArray1[dimensionFrameRecord-1] = 0 
        
        summFrame = runGoodArray1.sum(dtype=np.int32)
        if summFrame <= minFrame_1Set:
            Obj1_Adder = 1
        if summFrame >= maxFrame_1Add:
            Obj1_Total = Obj1_Total + Obj1_Adder
            Obj1_Adder = 0
            
        textShow = "M5x15 : "+str(Obj1_Total) 
        cv2.putText(resized, textShow, (100,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
        
        #-----------------------------------------------------------------------
        if len(kp2) >= minGoodPoint_A:
            runGoodArrayA[dimensionFrameRecord-1] = 1  
        else:
            runGoodArrayA[dimensionFrameRecord-1] = 0 
        
        summFrame = runGoodArrayA.sum(dtype=np.int32)
        if summFrame <= minFrame_ASet:
            ObjA_Adder = 1
        if summFrame >= maxFrame_AAdd:
            ObjA_Total = ObjA_Total + ObjA_Adder
            ObjA_Adder = 0
            
        textShow ="All : " + str(ObjA_Total)  
        cv2.putText(resized, textShow, (100,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 1)
        
        #-----------------------------------------------------------------------
        match_result = cv2.drawMatches(image1C, kp1, resized, kp2, good[:50], None, flags=2)
        cv2.imshow('Frame',match_result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()
print("Total = ",ObjA_Total ,", Object Type_1 = ", Obj1_Total)