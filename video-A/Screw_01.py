from reprlib import aRepr
import cv2
import numpy as np

cap = cv2.VideoCapture("./IMG_8669.MOV")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

scale_percent = 40
width = int(int(cap.get(3)) * scale_percent/100)
height = int(int(cap.get(4)) * scale_percent/100)
dim = (width,height)

size = (frame_width, frame_height)

result = cv2.VideoWriter('./name.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)

print(frame_width,frame_height)
print(width,height)
lower = np.array([116, 100, 100])                            
upper = np.array([127, 255, 255]) 


font = cv2.FONT_HERSHEY_SIMPLEX

m5x10_Counter = 0
m5x15_Counter = 0
m5x20_Counter = 0
m5x25_Counter = 0
m5x50_Counter = 0

stsCount = 0
stsLine_In = 1
stsLine_Out = 1

frameCount_50, delayFrame_50 = 1234, 15
frameCount_25, delayFrame_25 = 1234, 15
frameCount_20, delayFrame_20 = 1234, 15
frameCount_15, delayFrame_15 = 1234, 15
frameCount_10, delayFrame_10 = 1234, 15

while(cap.isOpened()):
    ret, frame = cap.read()
    resized = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

    roi = resized[150: 300, 0: 340]  
    if frame is None:
        break
    
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)                     
    mask = cv2.inRange(hsv, lower, upper)     
    res = cv2.bitwise_and(roi, roi, mask = mask) 

    # thresh = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,7)
    # cv2.imshow("thresh",thresh)

    # kernel = np.ones((1,1),np.uint8)
    # closing = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=10)
    # result_img = closing.copy()
    # cv2.imshow("closing",result_img)
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # #ref in
    cv2.line(resized, (200,150),(200,300),  (255,255,255), 2)
    # #ref out
    cv2.line(resized, (300,150),(300,300),  (255,255,255), 2)

    frameCount_50 += 1
    frameCount_25 += 1 
    frameCount_20 += 1
    frameCount_15 += 1
    frameCount_10 += 1

    for i,cntr in enumerate(contours):
        area = cv2.contourArea(cntr)
        x,y,w,h = cv2.boundingRect(cntr)
        if(area > 200 and area < 1700):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            # cv2.line(frame, (x,y),  (x+w,y+h),(0,255,0), 2)
            # cv2.line(frame, (x+w,y),(x,y+h),  (0,255,0), 2)

            cv2.rectangle(roi, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(roi,str(area), (x, 50), font, 1, (0 , 255 , 0), 4, cv2.LINE_AA)
#=======================M5x50=================================#
        if(area>1000 and area<1700):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (19,69,139), 2)
            if (x+w) > 200 :
                stsLine_In = 1
                cv2.line(resized, (200,150),(200,300),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) > 300 :
                stsLine_Out = 1
                cv2.line(resized, (300,150),(300,300),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount_50 > delayFrame_50):
                m5x50_Counter = m5x50_Counter + stsCount
                frameCount_50 = 0
                stsCount = 0
#=======================M5x50=================================#
#=======================M5x25=================================#
        if(area>600 and area<=700):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (255,255,0), 2)
            if (x+w) > 200 :
                stsLine_In = 1
                cv2.line(resized, (200,150),(200,300),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) > 300 :
                stsLine_Out = 1
                cv2.line(resized, (300,150),(300,300),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount_25 > delayFrame_25):
                m5x25_Counter = m5x25_Counter + stsCount
                frameCount_25 = 0
                stsCount = 0
#=======================M5x25=================================#
#=======================M5x20=================================#
        if(area>500 and area<=600):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,0), 2)
            if (x+w) > 200 :
                stsLine_In = 1
                cv2.line(resized, (200,150),(200,300),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) > 300 :
                stsLine_Out = 1
                cv2.line(resized, (300,150),(300,300),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount_20 > delayFrame_20):
                m5x20_Counter = m5x20_Counter + stsCount
                frameCount_20 = 0
                stsCount = 0
#=======================M5x20=================================#
#=======================M5x15=================================#
        if(area>400 and area<=500):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (127,0,255), 2)
            if (x+w) > 200 :
                stsLine_In = 1
                cv2.line(resized, (200,150),(200,300),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) > 300 :
                stsLine_Out = 1
                cv2.line(resized, (300,150),(300,300),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount_15 > delayFrame_15):
                m5x15_Counter = m5x15_Counter + stsCount
                frameCount_15 = 0
                stsCount = 0
#=======================M5x15=================================#
#=======================M5x10=================================#
        if(area>350 and area<=400):
            # cv2.drawContours(roi, [cntr], -1, (0, 255, 0), 2)
            cv2.rectangle(roi, (x,y), (x+w,y+h), (0,255,255), 2)
            if (x+w) > 200 :
                stsLine_In = 1
                cv2.line(resized, (200,150),(200,300),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) > 300 :
                stsLine_Out = 1
                cv2.line(resized, (300,150),(300,300),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount_10 > delayFrame_10):
                m5x10_Counter = m5x10_Counter + stsCount
                frameCount_10 = 0
                stsCount = 0
#=======================M5x10=================================#


    cv2.rectangle(resized,(0,0),(800,150),(0,0,0),-1)

    cv2.putText(resized, "Total = "+str(m5x50_Counter+m5x25_Counter+m5x20_Counter+m5x15_Counter+m5x10_Counter), (50, 110), font, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.putText(resized, "M5x50 = "+str(m5x50_Counter), (50, 50), font, .5, (19,69,139), 1, cv2.LINE_AA)


    cv2.putText(resized, "M5x25 = "+str(m5x25_Counter), (200, 50), font, .5, (255 , 255 , 0), 1, cv2.LINE_AA)


    cv2.putText(resized, "M5x20 = "+str(m5x20_Counter), (350, 50), font, .5, (0 , 255 , 0), 1, cv2.LINE_AA)


    cv2.putText(resized, "M5x15 = "+str(m5x15_Counter), (500, 50), font, .5, (127,0,255), 1, cv2.LINE_AA)


    cv2.putText(resized, "M5x10 = "+str(m5x10_Counter), (650, 50), font, .5, (0,255,255), 1, cv2.LINE_AA)


    cv2.putText(resized,"stsLine_In : "+str(stsLine_In), (500, 370), font, .5, (0 , 0 , 0), 1, cv2.LINE_AA)
    cv2.putText(resized,"stsLine_Out : "+str(stsLine_Out), (500, 400), font, .5, (0 , 0 , 0), 1, cv2.LINE_AA)

    #result.write(frame)
    cv2.imshow('mask', mask)
    cv2.imshow('Roi', roi)
    cv2.imshow('re', resized)

#     cv2.imshow('Mask-', mask)
#     cv2.imshow('Res-', res_r)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC Key
        break
print(m5x10_Counter+m5x15_Counter+m5x20_Counter+m5x25_Counter+m5x50_Counter)
cap.release()
cv2.destroyAllWindows()