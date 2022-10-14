# Step 1/4 Mask Frame
import cv2
cap = cv2.VideoCapture("./video_B5.avi")
object_detector = cv2.createBackgroundSubtractorMOG2()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

result = cv2.VideoWriter('./Vedio_B_Result.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)

font = cv2.FONT_HERSHEY_SIMPLEX
nCounter = 0

stsCount = 0
stsLine_In = 0
stsLine_Out = 0

frameCount, delayFrame = 1234, 10

while(cap.isOpened()):
    ret, frame = cap.read()
    # 1. Object Detection
    mask = object_detector.apply(frame)
    # 2. Find Contours
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #ref in
    cv2.line(frame, (500,0),(500,1024),  (255,0,0), 2)
    #ref out
    cv2.line(frame, (400,0),(400,1024),  (255,0,0), 2)
    
    frameCount = frameCount + 1
    for cnt in contours:
      # Calculate area and remove small elements
      area = cv2.contourArea(cnt)
      if area > 100:
      #Show image
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            if (x+w) < 500 :
                stsLine_In = 1
                cv2.line(frame, (500,0),(500,1024),  (0,0,255), 2)
            else:
                stsLine_In = 0
            if (x+w) < 400 :
                stsLine_Out = 1
                cv2.line(frame, (400,0),(400,1024),  (0,0,255), 2)
            else:
                stsLine_Out = 0                   
            if(stsLine_Out==0 and stsLine_In==0):
                stsCount = 1           
            if(stsLine_Out==1 and stsLine_In==1 and frameCount > delayFrame):
                nCounter = nCounter + stsCount
                frameCount = 0
                stsCount = 0

            
    cv2.putText(frame, "Total = "+str(nCounter), (50, 110), font, 2, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RefIn : " + str(stsLine_In), (520, 1000), font, .5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(frame, "RefOut : " + str(stsLine_Out), (300, 1000), font, .5, (0,0,0), 1, cv2.LINE_AA)
    cv2.imshow("ResultFrame", frame)
    result.write(frame)
    
    
    

    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

