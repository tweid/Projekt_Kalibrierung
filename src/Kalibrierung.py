import cv2


def main():
    print("test")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    patternSize = (9, 6)

    imgLeft = cv2.imread("TI119_L.jpg")
    imgRight = cv2.imread("TI119_R.jpg")

    foundCornersLeft, leftCorners = cv2.findChessboardCorners(imgLeft, patternSize, None)
    foundCornersRight, rightCorners = cv2.findChessboardCorners(imgRight, patternSize, None)

    if foundCornersLeft == True & foundCornersRight == True:
        print("left corners:")
        print(leftCorners)
        print("\n\n\n\n\n")
        print("rigth corners:")
        print(rightCorners)

        grayLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(grayLeft, leftCorners, (11, 11), (-1, -1), criteria)

        imgLeft = cv2.drawChessboardCorners(imgLeft, patternSize, leftCorners, foundCornersLeft)

        cv2.imwrite("editedLeft.jpg", imgLeft)
        print("written")




    else:
        print("Didn't found whished corners")
        print("Found corners left:")
        print(foundCornersLeft)
        print("Found corners right:")
        print(foundCornersRight)


if __name__ == "__main__":
    main()
