import numpy as np
import cv2


def main():
    print("test")

    patternSize = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points array for size
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objPoints = np.zeros((patternSize[0]*patternSize[1],3), np.float32)
    objPoints[:,:2] = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objectPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.


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

        # refining image points
        grayLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(grayLeft, leftCorners, (11, 11), (-1, -1), criteria)

        imgLeft = cv2.drawChessboardCorners(imgLeft, patternSize, leftCorners, foundCornersLeft)
        cv2.imwrite("editedLeft.jpg", imgLeft)
        print("written")

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        # append for right format
        objectPoints.append(objPoints)
        imagePoints.append(leftCorners)
        reprojectionError, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv2.calibrateCamera(objectPoints, imagePoints, grayLeft.shape[::-1],None,None)

        print("\n\n\nCamera Matrix:")
        print(cameraMatrix)
        print("\n\nReprojection Error:")
        print(reprojectionError)




    else:
        print("Didn't found whished corners")
        print("Found corners left:")
        print(foundCornersLeft)
        print("Found corners right:")
        print(foundCornersRight)


if __name__ == "__main__":
    main()
