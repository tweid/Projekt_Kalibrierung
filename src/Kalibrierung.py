import numpy as np
import cv2
from scipy.sparse.linalg import svds, eigs


def affineReconstruction(corners, patternSize):
    # i = 1, ..., m --> Count of camera matrices and therefore images
    # j = 1, ..., n --> Count of realworld points and therefore imagepoints per image

    # computation of translation:
    # t^i = <x^i> = 1 / n * sum(x^i(j))
    centroid = sum(corners) / (patternSize[0]*patternSize[1])
    print("Centroid:")
    print(centroid)

    # centre the data:
    # x^i(j) <-- x^i(j) - t^i
    centredPoints = corners - centroid
    print("\n\nCentred Points")
    print(centredPoints[:,0])

    # Construct measurement matrix W
    #      _                            _
    #     | x1,1    x1,2    ...     x1,n |
    #     | x2,1    x2,2    ...     x2,n |
    # W = | ...     ...     ...     ...  |
    #     | xm,1    xm,2    ...     xm,n |
    #     ‾‾                            ‾‾
    #
    # n = Count of image/world points
    # m = Count of images / camera matrices
    # 2D-Points (x) stacked vertically
    # --> W = 2m*n matrix

    #todo: refactoring
    # measurement line to form x and y in different columns
    measurementLine = np.vstack((centredPoints[:,0,0], centredPoints[:,0,1]))
    measurementMatrix = np.vstack((measurementLine, measurementLine))
    print("\n\n\nMeasurement Matrix:")

    #compute svd
    u, dTemp, vT = svds(measurementMatrix, k=3)
    d = np.diag(dTemp)

    #todo: matrices in seperate arrays
    M = np.matmul(u, d)
    #X = np.matrix.getH(vT)
    X = vT
    #seems like (y, x, z)

    return M, centroid[0], X






def calculate3dTo2d(M, t, X):
    # (x, y) = M * (x, y, z) + t
    xyTemp = np.matmul(M, X)
    xy = []
    for i in range(len(xyTemp[0])):
        xy.append(xyTemp[:, i] + t)

    return xy








def main():
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

        # append for right format
        objectPoints.append(objPoints)
        imagePoints.append(leftCorners)
        reprojectionError, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv2.calibrateCamera(objectPoints, imagePoints, grayLeft.shape[::-1],None,None)

        print("\n\n\nCamera Matrix:")
        print(cameraMatrix)
        print("\n\nReprojection Error:")
        print(reprojectionError)






        # Affine reconstruction - factorization alorithm
        print("\n\n\nAffine Reconstruction")
        M, t, X = affineReconstruction(leftCorners, patternSize)


        print("\n\nCameraMatrizes:")
        print(M)
        print("\n3D-Points:")
        print(X)

        print(calculate3dTo2d(M[:2], t, X))








    else:
        print("Didn't found whished corners")
        print("Found corners left:")
        print(foundCornersLeft)
        print("Found corners right:")
        print(foundCornersRight)


if __name__ == "__main__":
    main()
