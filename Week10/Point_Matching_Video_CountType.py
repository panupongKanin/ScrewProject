import cv2
import numpy as np
nDimGoodArray = 20
refGoodPoint  = 20
refFrameCount = 10

sift = cv2.SIFT_create()

image1C = cv2.imread("./imgScrew7.png")
image1G = cv2.cvtColor(image1C, cv2.IMREAD_GRAYSCALE)  
kp1, des1 = sift.detectAndCompute(image1G, None)

cap = cv2.VideoCapture("./video_B5.avi")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('./Vedio_B_Result.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)


ret, frame = cap.read()
height, width = frame.shape[:2]
positionText = (30, int(0.20*height))    # 25%
runGoodArray1 = np.zeros(nDimGoodArray)
Obj1_Total, Obj1_Adder = 0, 0

while(cap.isOpened()):
    ret, image2C = cap.read()
    if ret == True:
        image2G = cv2.cvtColor(image2C, cv2.IMREAD_GRAYSCALE) 
        kp2, des2 = sift.detectAndCompute(image2G, None)
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
        

        for iShift in range(nDimGoodArray-1):
            runGoodArray1[iShift] = runGoodArray1[iShift+1]
        if len(good) > refGoodPoint:
            runGoodArray1[nDimGoodArray-1] = 1  
        else:
            runGoodArray1[nDimGoodArray-1] = 0 
        
        summFrame = runGoodArray1.sum(dtype=np.int32)
        if summFrame == 0:
            Obj1_Adder = 1
        if summFrame > 10:
            Obj1_Total = Obj1_Total + Obj1_Adder
            Obj1_Adder = 0
        
        textShowTotal = "M5x15 : "+str(Obj1_Total) 
        cv2.putText(image2C,textShowTotal ,(100,100), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0), 2)


      #   textShow = str(Obj1_Total) + "<" + str(len(good)) + "," + str(summFrame) + ">"    
      #   cv2.putText(image2C,textShow , positionText, cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 2)
           
        match_result = cv2.drawMatches(image1C, kp1, image2C, kp2, good[:50], None, flags=2)
        

    
        cv2.imshow('Frame',match_result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()
print("M5x20 : ", Obj1_Total)