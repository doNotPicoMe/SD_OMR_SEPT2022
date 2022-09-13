import cv2
import numpy as np

# IMPORT FUNCTIONS FROM UTILITY FILE
import utility

#########IMAGE#########
# path="sample.jpg"
path="FCUC_OMR.png"
widthImg=800
heightImg=800
#########IMAGE#########

##OMR QUESTION FORMAT##
questions=5
choices=5
ans = [1,2,0,1,4]
##OMR QUESTION FORMAT##

# RETRIEVE IMAGE
img=cv2.imread(path)

# PREPROCESSING
img=cv2.resize(img,(widthImg,heightImg))
imgContours=img.copy()
imgFinal=img.copy()
imgBiggestContours=img.copy()
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)

# FIND ALL CONTOURS
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# DISPLAY CONTOURS, -1 means all index,RGB Green Colour(0,255,0),thickness=10
cv2.drawContours(imgContours,contours,-1,(0,255,0),10)

# FIND RECTANGLES
rectCon=utility.rectContour(contours)

#BIGGEST CONTOUR (BUBBLE)
biggestContour=utility.getCornerPoints(rectCon[0])
# print(biggestContour.shape)
#2nd BIGGEST CONTOUR (RESULT ZONE)
gradePoints= utility.getCornerPoints(rectCon[1])

#LOCATE BUBBLE and RESULT ZONE
if biggestContour.size!=0 and gradePoints.size!=0:
    cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
    cv2.drawContours(imgBiggestContours,gradePoints,-1,(255,0,0),20)

    biggestContour=utility.reorder(biggestContour)
    gradePoints=utility.reorder(gradePoints)

    # 1. BIRDS EYE VIEW: BUBBLE ZONE
    pt1=np.float32(biggestContour)
    pt2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pt1,pt2)
    imgWarpColored=cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    # 2. BIRDS EYE VIEW: GRADE ZONE
    ptGrade1=np.float32(gradePoints)
    ptGrade2=np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixGrade=cv2.getPerspectiveTransform(ptGrade1,ptGrade2)
    imgGradeDisplay=cv2.warpPerspective(img,matrixGrade,(325,150))
    # cv2.imshow("Grade",imgGradeDisplay)

    #3. APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    # 170 determines the intensity of the shaded region
    imgThresh =cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

    # 4. FIND INDIVIDUAL BUBBLES
    boxes = utility.splitBoxes(imgThresh)
    # cv2.imshow("Test",boxes[2])

    # 5. FIND MARK BUBBLES (USE Non-Zero Pixels)
    # print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))

    # 6. GETTING NON-Zero PIXEL VALUES OF EACH BUBBLE
    myPixelValue=np.zeros((questions,choices))
    countCol=0
    countRow=0

    # Loop through all bubbles
    for image in boxes:
        totalPixels=cv2.countNonZero(image)
        # 5x5 array (5 questions and 5 choices)
        myPixelValue[countRow][countCol]=totalPixels
        countCol+=1
        if(countCol==choices):countRow+=1;countCol=0
    # print(myPixelValue)


    # FINDING INDEX VALUES OF THE MARKING
    # LOGIC: USE THE MAXIMUM VALUE, TO GET THE SHADED BUBBLE
    myIndex = []
    for x in range (0,questions):
        arr= myPixelValue[x]
        # print("arr",arr)
        myIndexVal=np.where(arr==np.amax(arr))
        # print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)

    # GRADING
    grading=[]
    for x in range (0,questions):
        if ans[x]==myIndex[x]:
            # 1 if answer is correct, 0 if answer is wrong
            grading.append(1)
        else: grading.append(0)
    # print(grading)
    score = sum(grading)/questions *100 # FINAL GRADE
    print(score)

    #DISPLAY ANSWERS
    imgResults = imgWarpColored.copy()
    imgResults = utility.showAnswers(imgResults,myIndex, grading, ans, questions, choices)

    #DISPLAY ANSWERS (Just shades only)
    imgRawDrawing = np.zeros_like(imgWarpColored)
    imgRawDrawing = utility.showAnswers(imgRawDrawing,myIndex,grading,ans,questions,choices)

    #DISPLAY ANSWERS (Based on original image)
    invMatrix = cv2.getPerspectiveTransform(pt2,pt1)
    imgInvWarp = cv2.warpPerspective(imgRawDrawing,invMatrix,(widthImg,heightImg))

    #DISPLAY GRADES (Based on original image)
    imgRawGrade = np.zeros_like(imgGradeDisplay)
    cv2.putText(imgRawGrade,str(int(score))+"%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
    # cv2.imshow("Grade",imgRawGrade)
    invMatrixG = cv2.getPerspectiveTransform(ptGrade2,ptGrade1)
    # imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (325,150))
    imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg,heightImg))
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvWarp,1,0)
    imgFinal = cv2.addWeighted(imgFinal,1,imgInvGradeDisplay,1,0)

# print(len(biggestContour))
# print(biggestContour)

# BLANK IMAGE DECLARATION
imgBlank=np.zeros_like(img)

# DISPLAY AN ARRAY OF IMAGE
imageArray = ([img,imgGray,imgBlur,imgCanny],
              [imgContours,imgBiggestContours,imgWarpColored,imgThresh],
              [imgResults,imgRawDrawing,imgInvWarp,imgFinal])

labels = [["Original","Gray","Blur","Blur"],
          ["Contours","Biggest Contours","Warp","Threshold"],
          ["Result","Raw","Inverse","Inverse Warp","Final"]
          ]
imgStacked = utility.stackImages(imageArray,0.4,labels)
# DISPLAY FINAL IMAGE
cv2.imshow("Stacked Images",imgFinal)
cv2.imshow("Stacked Images",imgStacked)


# TEST OPENCV OUTPUT: VIEW OUTPUT
# cv2.imshow("Original",img)
cv2.waitKey(0)
