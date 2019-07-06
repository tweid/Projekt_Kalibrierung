import time

import numpy as np
import cv2
from scipy.sparse.linalg import svds, eigs
from scipy.spatial import distance
import glob

PATTERN_SIZE = (9, 6)

CONTROLLING = True #Is a Controlling-Pattern there
HORIZONTAL_PATTERN_DISTANCE = 1 #Horizontal distance between Pattern sheets in meter (<0: control pattern left, >0: right)
VERTICAL_PATTERN_DISTANCE = 0.05 #Vertical distance between Pattern sheets in meter (<0: control pattern up, >0: down)

POINT_DISTANCE = 0.02 #Distance between points in meter

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

CALC_FILE_BEGINNING = 'calc'
CONTROL_FILE_BEGINNING = 'control'
JPG_NAME = CALC_FILE_BEGINNING + '*.jpg'


def affineReconstruction(corners, patternSize):
    # i = 1, ..., m --> Count of camera matrices and therefore images
    # j = 1, ..., n --> Count of realworld points and therefore imagepoints per image
    centroid = []
    for i in range(len(corners)):
        # computation of translation:
        # t^i = <x^i> = 1 / n * sum(x^i(j))
        centroid.append(sum(corners[i]) / (patternSize[0]*patternSize[1]))
    #print("Centroid:")
    centroid = np.reshape(centroid, (len(corners), 2))
    #print(centroid)

    # centre the data:
    # x^i(j) <-- x^i(j) - t^i
    centredPoints = []
    for i in range(len(corners)):
        centredPoints.append(corners[i] - centroid[i])
    #print("\n\nCentred Points")
    #print(centredPoints)

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

    measurementMatrix = []
    for i in range(len(corners)):
        measurementMatrix.append(np.vstack((centredPoints[i][:, 0, 0], centredPoints[i][:, 0, 1])))
    measurementMatrix = np.reshape(measurementMatrix, (len(corners)*2, len(centredPoints[0])))
    #print("\n\n\nMeasurement Matrix:\n", measurementMatrix)

    #compute svd
    u, dTemp, vT = svds(measurementMatrix, k=3)
    d = np.diag(dTemp)

    M = np.matmul(u, d)
    #X = np.matrix.getH(vT)
    X = vT
    #seems like (y, x, z)

    return M, centroid, X













def calculate3dTo2d(M, t, X):
    # (x, y) = M * (x, y, z) + t
    xyTemp = np.matmul(M, X)
    xy = []
    for i in range(len(xyTemp[0])):
        xy.append(xyTemp[:, i] + t)

    return xy











def affineReprojection(affineCameraMatrizes, imagePoints, t, X):
    affineReprojectedPoints = []
    affineReprojectionError = []
    for i in range(len(affineCameraMatrizes)):
        affineReprojectedPoints.append(calculate3dTo2d(affineCameraMatrizes[i], t[i], X))
        affineReprojectionError.append(computeMeanReprojectionError(imagePoints[i], affineReprojectedPoints[i][:]))
    affineReprojectedPoints = np.reshape(affineReprojectedPoints, (len(affineCameraMatrizes), len(imagePoints[0]), 2))
    return affineReprojectedPoints, affineReprojectionError














def computeMeanReprojectionError(imgPoints, reprojectedPoints):
    #Alternate Method: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    #mean_error = 0
    #for i in range(len(objectPoints)):
    #    imgpoints2, _ = cv2.projectPoints(objectPoints[i], rotationVecs[i], translationVecs[i], cameraMatrix, distCoeffs)
    #    error = cv2.norm(imagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #    mean_error += error
    #    print( "total error: {}".format(mean_error/len(objectPoints)) )

    #imgPoints: [[[x0,0 y0,0]] [[x0,1 y0,1]]]]
    #https://stackoverflow.com/questions/23781089/opencv-calibratecamera-2-reprojection-error-and-custom-computed-one-not-agree
    total_error=0
    total_points=0
    for i in range(len(imgPoints)):
        err = 0;
        total_points += len(imgPoints[i])
        p = imgPoints[i][0]
        q = reprojectedPoints[i]
        err += (p[0] - q[0])**2 + (p[1] - q[1])**2
        total_error += err

    mean_error=np.sqrt(total_error/total_points)
    return mean_error
























def movingAffine(X, affineCameraMatrizes, foundCorners, images, t):
    # 3d-Points like y, x, z.
    # y-difference = 0.001 und alle 9 0.8
    # x-difference = 0.5
    # z-difference = random
    relativeDistance = [VERTICAL_PATTERN_DISTANCE / POINT_DISTANCE, HORIZONTAL_PATTERN_DISTANCE / POINT_DISTANCE]  # Ratio between Distance of Patterns and Distance of Points
    newX = X.copy()
    relative3dDistance = [0, 0, 0]

    #Horizontal Distance
    for i in range(PATTERN_SIZE[1]):  # For each row
        # Sum up Distance between Points
        relative3dDistance[1] += X[1][i * PATTERN_SIZE[0]] - X[1][(i + 1) * PATTERN_SIZE[0] - 1]
    # Get mean Distance by dividing by number of added Distances
    relative3dDistance[1] = relative3dDistance[1] / (PATTERN_SIZE[1] * (PATTERN_SIZE[0] - 1))
    # Multiplying with relative Distance
    relative3dDistance[1] = relative3dDistance[1] * relativeDistance[1]

    #Vertical Distance
    for i in range(PATTERN_SIZE[0]): #For each column
        #Sum of vertical distances
        relative3dDistance[0] += X[0][i] - X[0][i + PATTERN_SIZE[0] * (PATTERN_SIZE[1] - 1)]
    # Get mean Distance by dividing by number of added Distances
    relative3dDistance[0] = relative3dDistance[0] / ((PATTERN_SIZE[0] - 1) * PATTERN_SIZE[1])
    # Multiplying with relative Distance
    relative3dDistance[0] = relative3dDistance[0] * relativeDistance[0]


    newX = np.add(newX, np.reshape(relative3dDistance, (3, 1)))

    reprojectionError = []
    for imageNumber in range(len(images)):
        fileName = images[imageNumber].replace(CALC_FILE_BEGINNING, CONTROL_FILE_BEGINNING)
        newImagePoints = calculate3dTo2d(affineCameraMatrizes[imageNumber], t[imageNumber], newX)
        newImagePoints = np.reshape(newImagePoints, (PATTERN_SIZE[0] * PATTERN_SIZE[1], 1, 2))
        image = cv2.imread(fileName)

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foundCorners, corners = cv2.findChessboardCorners(grayImage, PATTERN_SIZE, None)
        cornersRefined = cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1), CRITERIA)
        reprojectionError.append(computeMeanReprojectionError(cornersRefined, np.reshape(newImagePoints, (PATTERN_SIZE[0]*PATTERN_SIZE[1], 2))))

        image = cv2.drawChessboardCorners(image, PATTERN_SIZE, newImagePoints.astype(np.float32), foundCorners)
        cv2.imwrite("edited_" + fileName, image)

    print("\n\nControl image reprojection error:\n", reprojectionError)


























def main():
    # prepare object points array for size
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objPoints = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3), np.float32)
    objPoints[:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objectPoints = [] # 3d point in real world space
    imagePoints = [] # 2d points in image plane.

    images = glob.glob(JPG_NAME)

    for fileName in images:
        print("Processing ", fileName)
        image = cv2.imread(fileName)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foundCorners, corners = cv2.findChessboardCorners(grayImage, PATTERN_SIZE, None)

        if foundCorners == True:
            print("Found Corners!")
            cornersRefined = cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1), CRITERIA)
            image = cv2.drawChessboardCorners(image, PATTERN_SIZE, cornersRefined, foundCorners)
            cv2.imwrite("edited_" + fileName, image)

            objectPoints.append(objPoints)
            imagePoints.append(cornersRefined)
            print("Image Points processed with OpenCV\n\n")


        else:
            print("Didn't found corners in ", fileName)

    startOpenCV = time.time()
    reprojectionError, cameraMatrix, distCoeffs, rotationVecs, translationVecs = cv2.calibrateCamera(objectPoints, imagePoints, grayImage.shape[::-1],None,None)
    endOpenCV = time.time()
    timeOpenCV = endOpenCV - startOpenCV

    print("\n\n\nCamera Matrix:")
    print(cameraMatrix)
    print("\n\nOpenCV Reprojection Error:")
    print(reprojectionError)


    meanReprojectionErrors = []
    allReprojectedPoints = []
    for i in range(len(objectPoints)):
        reprojectedPoints, _ = cv2.projectPoints(objectPoints[i], rotationVecs[i], translationVecs[i], cameraMatrix, distCoeffs)
        reprojectedPoints = reprojectedPoints.reshape(-1,2)
        allReprojectedPoints.append(reprojectedPoints)
        meanReprojectionErrors.append(computeMeanReprojectionError(imagePoints[i], reprojectedPoints))
    print("\n\n\nReprojected Points:\n", allReprojectedPoints);
    print("\nSelfcalculated Reprojection Errors:\n", meanReprojectionErrors)






    # Affine reconstruction - factorization alorithm
    print("\n\n\nAffine Reconstruction")
    startAffine = time.time()
    M, t, X = affineReconstruction(imagePoints, PATTERN_SIZE)
    endAffine = time.time()
    timeAffine = endAffine - startAffine
    affineCameraMatrizes = np.split(M, len(images))


    print("\n\nAffine CameraMatrizes:")
    print(M)
    print("\n\nAffine CameraMatrizes:")
    print(affineCameraMatrizes)
    print("\n3D-Points:")
    print(X)
    print("\nt:\n", t)

    affineReprojectedPoints, affineReprojectionError = affineReprojection(affineCameraMatrizes, imagePoints, t, X)
    print("\n\n\nAffine Reprojected Points;\n", affineReprojectedPoints)
    print("\n\nAffine Reprojection Error:\n", affineReprojectionError)
    print("\n\nTime needed by OpenCV: ", timeOpenCV)
    print("Time needed by Affine: ", timeAffine)

    if CONTROLLING:
        movingAffine(X, affineCameraMatrizes, foundCorners, images, t)

if __name__ == "__main__":
    main()
