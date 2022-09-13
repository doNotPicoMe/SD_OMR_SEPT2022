import cv2
import numpy as np
from PIL import Image

# IMPORT FUNCTIONS FROM utility_long_2 FILE
import utility_long_2

#IMAGE#
path= "OMR_2.png"
# ADJUST HERE TO IMRPOVE CLARITY 600:800, 3:4
widthImg=800    #BEST RATIO FOR OMR.png
heightImg=1000  #BEST RATIO FOR OMR.png
#IMAGE#

#OMR QUESTION FORMAT#
questions=20
choices=4
# FULL MARKS FOR OMR.png
# ans = [0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,
#        0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,
#        0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1
#        ]
# FULL MARKS FOR OMR_2.png
# ans = [3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,
#        0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,
#        0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1
#        ]

ans ={}
ans[0]=(3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2)
ans[1]=(3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2)
ans[2]=(3,2,1,0,1,2,3,2,1,0,1,2,3,2,1,0,1,2,3,2)
ansCounter=0
score=0
bubbleList=[None]*3     #3 will be change based on the number of bubble columns
subScore =[None]*3      #3 will be change based on the number of bubble columns
imgResults =[None]*3    #3 will be change based on the number of bubble columns
imgRawDrawing =[None]*3 #3 will be change based on the number of bubble columns
imgInvWarp =[None]*3     #3 will be change based on the number of bubble columns
invMatrix =[None]*3     #3 will be change based on the number of bubble columns
#OMR QUESTION FORMAT#


# IMAGE PREPROCESSING #
img=cv2.imread(path)                            # RETRIEVE IMAGE
img=cv2.resize(img,(widthImg,heightImg))        # RESIZE IMAGE
imgContours=img.copy()                          # CREATE A COPY OF ORIGINAL IMAGE
imgFinal=img.copy()                             # CREATE A COPY OF ORIGINAL IMAGE
imgBiggestContours=img.copy()                   # CREATE A COPY OF ORIGINAL IMAGE
# LOGIC: ORIGINAL → GRAY → BLUR → CANNY
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    # CONVERT ORIGINAL → GRAY
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)       # CONVERT GRAY → BLUR
imgCanny=cv2.Canny(imgBlur,10,50)               # CONVERT BLUR → CANNY
# IMAGE PREPROCESSING #

# 1. FIND ALL CONTOURS
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# 1.1. DISPLAY CONTOURS, -1 means all index,RGB Green Colour(0,255,0),thickness=10
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

# 2. FIND RECTANGLES
rectCon=utility_long_2.rectContour(contours)

# 3. FIND BUBBLE ZONE, coordinates FROM LEFT TO RIGHT [2],[1],[0]
colCounter=0
while colCounter<3:
    bubbleList[colCounter]=utility_long_2.getCornerPoints(rectCon[colCounter])
    colCounter +=1

# 4. FIND RESULT ZONE
gradePoints= utility_long_2.getCornerPoints(rectCon[3])

scanCounter=0
while scanCounter<3:
    # 5. LOCATE BUBBLE and RESULT ZONE
    if bubbleList[scanCounter].size!=0 and gradePoints.size!=0:
        cv2.drawContours(imgBiggestContours,bubbleList[scanCounter],-1,(0,255,0),20)
        cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

        bubbleList[scanCounter]=utility_long_2.reorder(bubbleList[scanCounter])
        gradePoints=utility_long_2.reorder(gradePoints)

        # 6. BIRDS EYE VIEW: BUBBLE ZONE
        pt1=np.float32(bubbleList[scanCounter])
        pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix=cv2.getPerspectiveTransform(pt1,pt2)
        imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))
        # cv2.imshow("Bubble Zone",imgWarpColored) # VIEW RESULTS

        # TEMP. BIRDS EYE VIEW: GRADE ZONE [MIGHT HAVE TO RELOCATE!]
        ptGrade1 = np.float32(gradePoints)
        ptGrade2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixGrade = cv2.getPerspectiveTransform(ptGrade1, ptGrade2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixGrade, (325, 150))
        # cv2.imshow("Grade",imgGradeDisplay)   # VIEW RESULTS

        # TEMP. APPLY THRESHOLD [MIGHT HAVE TO RELOCATE!]
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        # 170, the parameter location controls the intensity of the shaded region
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        # 7. FIND INDIVIDUAL BUBBLES
        boxes = utility_long_2.splitBoxes(imgThresh)
        # cv2.imshow("Test",boxes[2])  # VIEW RESULTS

        # 8. GETTING NON-Zero PIXEL VALUES OF EACH BUBBLE
        bubblePixelValue = np.zeros((questions, choices))
        countCol = 0
        countRow = 0

        # 9. Loop through all bubbles
        for image in boxes:
            totalPixels = cv2.countNonZero(image)
            # 5x5 array (5 questions and 5 choices)
            bubblePixelValue[countRow][countCol] = totalPixels
            countCol += 1
            if (countCol == choices): countRow += 1;countCol = 0
        # print(bubblePixelValue) # VIEW RESULTS

        # 10. FINDING INDEX VALUES OF THE MARKING, LOGIC: USE THE MAXIMUM VALUE, TO GET THE SHADED BUBBLE
        bubbleIndex = []
        for x in range(0, questions):
            arr = bubblePixelValue[x]
            print("Q", x, ":", arr)
            bubbleIndexVal = np.where(arr == np.amax(arr))
            print("Shaded:", bubbleIndexVal[0])
            bubbleIndex.append(bubbleIndexVal[0][0])
            # print(bubbleIndex) # VIEW RESULTS

        # 11. GETTING THE SCORE OF EACH BUBBLE COLUMN,THEN SEND TO 12. FOR FINAL PROCESSING
        grading=[]
        for x in range (0,questions):
            if ans[x]==bubbleIndex[x]: # 1 if answer is correct, 0 if answer is wrong
                grading.append(1)
            else:
                grading.append(0)

        # print(grading) # VIEW RESULTS
        subScore[scanCounter] = sum(grading)/questions*100 # Sub-Grade
        print("SUBSCORE {",scanCounter,"} : ",subScore[scanCounter]) # VIEW RESULTS

    scanCounter += 1 # LOOP COUNTER

####################################

# 12. SUMMING ALL SCORES FROM EACH BUBBLE COLUMN
for i in range(0,len(subScore)):
    score=score+subScore[i];
    print("SCORE {",i,"} : ",subScore[i]) # VIEW RESULTS

score = (score/(3*100))*100 #3 will be change based on the number of bubble columns
print("SUM: ",float("{:.2f}".format(score))) # VIEW RESULTS, print 2 decimal places only
# print("SUM: ",score) # VIEW RESULTS

scanCounter = 0  # RESET LOOP COUNTER
while scanCounter<3:
    #DISPLAY ANSWERS
    imgResults[scanCounter] = imgWarpColored.copy()
    imgResults[scanCounter] = utility_long_2.showAnswers(imgResults[scanCounter],bubbleIndex, grading, ans, questions, choices)

    #DISPLAY ANSWERS (Just shades only)
    imgRawDrawing[scanCounter] = np.zeros_like(imgWarpColored)
    imgRawDrawing[scanCounter] = utility_long_2.showAnswers(imgRawDrawing[scanCounter],bubbleIndex,grading,ans,questions,choices)
    # cv2.imshow("Raw Drawing",imgRawDrawing) # VIEW RESULTS

    #DISPLAY ANSWERS (Based on original image), GET THIS TO WORK ONLY ONCE
    invMatrix[scanCounter] = cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWarp[scanCounter]= cv2.warpPerspective(imgRawDrawing[scanCounter],invMatrix[scanCounter],(widthImg,heightImg))
    # cv2.imshow("Inverse",imgInvWarp) # VIEW RESULTS

    scanCounter += 1  # LOOP COUNTER

# DISPLAY GRADES (Based on original image), GET THIS TO WORK ONCE!
imgRawGrade = np.zeros_like(imgGradeDisplay)
cv2.putText(imgRawGrade, str(int(score)) + "%", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
# cv2.imshow("Grade",imgRawGrade) # VIEW RESULTS
invMatrixG= cv2.getPerspectiveTransform(ptGrade2, ptGrade1)
# imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (325,150))
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

scanCounter = 0  # RESET LOOP COUNTER

# CONTINUE FROM HERE!
tempFinal=[None]*3
while scanCounter<3:
    # imgFinal = cv2.addWeighted(imgFinal, 0.8, imgInvWarp[scanCounter], 1, 0)
    tempFinal[scanCounter] = cv2.addWeighted(imgFinal, 0.8, imgInvWarp[scanCounter], 1, 0)
    cv2.imshow("FINAL IMAGE",tempFinal[scanCounter]) # VIEW RESULTS
    scanCounter +=1
# CONTINUE FROM HERE!

# imgFinal= cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)
# imgFinal= cv2.addWeighted(tempFinal, 1, imgInvGradeDisplay, 1, 0)
# cv2.imshow("FINAL IMAGE",imgFinal) # VIEW RESULTS

################################

# DISPLAY AN ARRAY OF IMAGE
imgBlank=np.zeros_like(img) # BLANK IMAGE DECLARATION

# imageArray = ([img,imgGray,imgBlur,imgCanny],
#               [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
#               [imgResults,imgRawDrawing,imgInvWarp,imgFinal])

imageArray = ([imgResults[0],imgRawDrawing[0],imgInvWarp[0],imgFinal],
              [imgResults[1],imgRawDrawing[1],imgInvWarp[1],imgFinal],
              [imgResults[2],imgRawDrawing[2],imgInvWarp[2],imgFinal])

labels = [["Original","Gray","Blur","Blur"],
          ["Contours","Biggest Contours","Warp","Threshold"],
          ["Result","Raw","Inverse","Inverse Warp","Final"]
          ]

imgStacked = utility_long_2.stackImages(imageArray,0.4,labels)

# TEST OPENCV OUTPUT: VIEW OUTPUT
# cv2.imwrite("FinalResult.png", imgFinal)   # DISPLAY IMAGE (Final)
# cv2.imshow("Stacked Images",imgResults)    # DISPLAY IMAGE (Results)
# cv2.imshow("Final Image", imgFinal)        # DISPLAY IMAGE (Final)
cv2.imshow("Stacked Images",imgStacked)      # DISPLAY IMAGE (Stack)

cv2.waitKey(0)

