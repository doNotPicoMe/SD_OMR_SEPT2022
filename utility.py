import cv2
import numpy as np

# 1. STACK ALL THE IMAGES IN ONE WINDOW (FIXED LOGIC)
def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

# 2. IDENTIFY RECTANGLE CONTOURS
def rectContour(contours):

    rectCon = []     #CREATE NEW LIST
    max_area = 0

    # 2.1. LOOP THROUGH ALL CONTOURS
    for i in contours:
        area = cv2.contourArea(i)
        # print(area) # CHECK CONTOUR AREA

        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("Corner Points: ", len(approx))
            if len(approx) == 4:
                rectCon.append(i)
                # print(rectCon)
    # 2.2. REORDER AND SORT RECTCON, BASED ON OMR FORM
    # rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True) # reverse=True, sort in descending order (biggest to smallest)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True) # reverse=True, sort in descending order (biggest to smallest)
    # print(len(rectCon))
    return rectCon

# 3. TBA
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    # print(np.argmax(add))
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]
    # print(diff)
    return myPointsNew

# 4. SPLIT BUBBLES VERTICALLY INSIDE OF ANSWER ZONE
def splitBoxes(img):

    rows = np.vsplit(img,5)     # 10, because there is 10 rows
    # cv2.imshow("Split",rows[0])

    # Split bubbles horizontally
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,4)   # 4, because there is 4 questions per column
        for box in cols:
            boxes.append(box)
            # cv2.imshow("Split",box)
    return boxes

# 5. GET CORNER POINTS OF CONTOUR
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

# 6. DISPLAY ANSWERS
# def showAnswers(img,myIndex,grading,ans,questions=5,choices=5): # (PREDEFINED)
def showAnswers(img,myIndex,grading,ans,questions,choices):

    # SIMULATE A 5*5 shape (based on answer zone)
    sectWidth = int(img.shape[1]/questions)
    sectHeight = int(img.shape[0]/choices)

    for x in range(0,questions):
        myAns = myIndex[x]

        # c = center of position
        # cX = (myAns * sectWidth) + sectWidth //2 #DEFAULT FORMAT
        # cY = (x * sectHeight) + sectHeight // 2 #DEFAULT FORMAT
        # print("cX=",cX," cY=",cY) #CHECK VALUES
        # print("Grading=",grading[x]) #CHECK VALUES
        cY = (x * sectWidth)+sectWidth//2

        if grading[x]==1:          # LOGIC: 1 correct answer, 0 wrong answer
            if myAns==0:
                cX = (myAns * sectWidth) + sectWidth // 2 + 50   #DEFAULT FORMAT
            elif myAns==1:
                cX = (myAns * sectWidth) + sectWidth // 2 + 80   #DEFAULT FORMAT
            elif myAns==2:
                cX = (myAns * sectWidth) + sectWidth // 2 + 110  #DEFAULT FORMAT
            elif myAns==3:
                cX = (myAns * sectWidth) + sectWidth // 2 + 140  # DEFAULT FORMAT

            myColor = (0,255,0)     # GREEN
            # myColor = (28,184,255)     # GREEN
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)  # Fill correct answer in GREEN

        elif grading[x]==0:
            if myAns == 0:
                cX = (myAns * sectWidth) + sectWidth // 2 + 50    #DEFAULT FORMAT
            elif myAns == 1:
                cX = (myAns * sectWidth) + sectWidth // 2 + 80    #DEFAULT FORMAT
            elif myAns == 2:
                cX = (myAns * sectWidth) + sectWidth // 2 + 110   #DEFAULT FORMAT
            elif myAns == 3:
                cX = (myAns * sectWidth) + sectWidth // 2 + 140   #DEFAULT FORMAT

            myColor = (0,255,0)     # GREEN
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img,(cX, cY),50,myColor,cv2.FILLED) # Fill correct answer in RED

            correctAns = ans[x]
            if correctAns== 0:
                # cX = (correctAns * sectWidth) + sectWidth // 2 + 50  # DEFAULT FORMAT
                cX = (correctAns * sectWidth) + sectWidth // 2 + 40  # TESTING
            elif correctAns== 1:
                # cX = (correctAns* sectWidth) + sectWidth // 2 + 80  # DEFAULT FORMAT
                cX = (correctAns* sectWidth) + sectWidth // 2 + 70  # TESTING
            elif correctAns == 2:
                # cX = (correctAns* sectWidth) + sectWidth // 2 + 110  # DEFAULT FORMAT
                cX = (correctAns* sectWidth) + sectWidth // 2 + 100  # TESTING
            elif correctAns == 3:
                # cX = (correctAns* sectWidth) + sectWidth // 2 + 140  # DEFAULT FORMAT
                cX = (correctAns* sectWidth) + sectWidth // 2 + 130  # TESTING

            myColor = (123,24,24)     # DARK RED
            # cv2.circle(img,((correctAns * sectWidth)+sectWidth//2, (x * sectHeight)+sectHeight//2), 20,myColor,cv2.FILLED) # Fill correct answer in GREEN
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)  # Fill correct answer in GREEN
        # else:
        #    None
    cv2.imshow("Show Answers",img)
    return img

# def drawGrid(img,questions=5,choices=5):
#     secW = int(img.shape[1]/questions)
#     secH = int(img.shape[0]/choices)
#     for i in range (0,9):
#         pt1 = (0,secH*i)
#         pt2 = (img.shape[1],secH*i)
#         pt3 = (secW * i, 0)
#         pt4 = (secW*i,img.shape[0])
#         cv2.line(img, pt1, pt2, (255, 255, 0),2)
#         cv2.line(img, pt3, pt4, (255, 255, 0),2)
#
#     return img

