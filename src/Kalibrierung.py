import time

import numpy as np
import cv2
from scipy.sparse.linalg import svds, eigs
from scipy.spatial import distance
import glob

PATTERN_SIZE = (9, 6)

CONTROLLING = True #Is a Controlling-Pattern there
HORIZONTAL_PATTERN_DISTANCE = 1 #Horizontal distance between Pattern sheets in meter (<0: control pattern left, >0: right)
VERTICAL_PATTERN_DISTANCE = 0.00 #Vertical distance between Pattern sheets in meter (<0: control pattern up, >0: down)

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

    M1 = np.matmul(u, d)
    M2 = u
    #X = np.matrix.getH(vT)
    X1 = vT
    X2 = np.matmul(d, vT)
    #seems like (y, x, z)

    return M1, X1, centroid, M2, X2













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
























def movingAffine(X, affineCameraMatrizes, images, t, addition):
    # 3d-Points like y, x, z.
    relativeDistance = [VERTICAL_PATTERN_DISTANCE / POINT_DISTANCE, HORIZONTAL_PATTERN_DISTANCE / POINT_DISTANCE]  # Ratio between Distance of Patterns and Distance of Points
    newX = X.copy()
    relative3dDistance = [0, 0, 0]
    relativeHorizontal3dDistance = [0, 0, 0]
    relativeVertical3dDistance = [0, 0, 0]

    #Horizontal Distance
    for j in range(len(relative3dDistance)):
        for i in range(PATTERN_SIZE[1]):  # For each row
            # Sum up Distance between Points
            relativeHorizontal3dDistance[j] += X[j][i * PATTERN_SIZE[0]] - X[j][(i + 1) * PATTERN_SIZE[0] - 1]
        # Get mean Distance by dividing by number of added Distances
        relativeHorizontal3dDistance[j] = relativeHorizontal3dDistance[j] / (PATTERN_SIZE[1] * (PATTERN_SIZE[0] - 1))
        # Multiplying with relative Distance
        relativeHorizontal3dDistance[j] = relativeHorizontal3dDistance[j] * relativeDistance[1]

    #Vertical Distance
    for j in range(len(relative3dDistance)):
        for i in range(PATTERN_SIZE[0]): #For each column
            #Sum of vertical distances
            relativeVertical3dDistance[j] += X[j][i] - X[j][i + PATTERN_SIZE[0] * (PATTERN_SIZE[1] - 1)]
        # Get mean Distance by dividing by number of added Distances
        relativeVertical3dDistance[j] = relativeVertical3dDistance[j] / ((PATTERN_SIZE[0] - 1) * PATTERN_SIZE[1])
        # Multiplying with relative Distance
        relativeVertical3dDistance[j] = relativeVertical3dDistance[j] * relativeDistance[0]

    relative3dDistance = np.add(relativeVertical3dDistance, relativeHorizontal3dDistance)
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
        cv2.imwrite("edited" + addition + "_" + fileName, image)

    print("\n\nControl image affine reprojection error" + addition + ":\n", reprojectionError)
    print("Mean: ", np.mean(reprojectionError))
























def movingCV(objectPoints, images, t, rotationVecs, translationVecs, cameraMatrix, distCoeffs):
    # 3d-Points like x, y, z.
    relativeDistance = [HORIZONTAL_PATTERN_DISTANCE / POINT_DISTANCE, VERTICAL_PATTERN_DISTANCE / POINT_DISTANCE]  # Ratio between Distance of Patterns and Distance of Points
    newObjectPoints = objectPoints.copy()
    relative3dDistance = [0, 0, 0]

    #Horizontal Distance
    for i in range(PATTERN_SIZE[1]):  # For each row
        # Sum up Distance between Points
        relative3dDistance[0] += objectPoints[i * PATTERN_SIZE[0]][0] - objectPoints[(i + 1) * PATTERN_SIZE[0] - 1][0]
    # Get mean Distance by dividing by number of added Distances
    relative3dDistance[0] = relative3dDistance[0] / (PATTERN_SIZE[1] * (PATTERN_SIZE[0] - 1))
    # Multiplying with relative Distance
    relative3dDistance[0] = relative3dDistance[0] * relativeDistance[0]

    #Vertical Distance
    for i in range(PATTERN_SIZE[0]): #For each column
        #Sum of vertical distances
        relative3dDistance[1] += objectPoints[i][1] - objectPoints[i + PATTERN_SIZE[0] * (PATTERN_SIZE[1] - 1)][1]
    # Get mean Distance by dividing by number of added Distances
    relative3dDistance[1] = relative3dDistance[1] / ((PATTERN_SIZE[0] - 1) * PATTERN_SIZE[1])
    # Multiplying with relative Distance
    relative3dDistance[1] = relative3dDistance[1] * relativeDistance[1]

    newObjectPoints = np.add(newObjectPoints, np.reshape(relative3dDistance, (1, 3)))


    reprojectionError = []
    for imageNumber in range(len(images)):
        fileName = images[imageNumber].replace(CALC_FILE_BEGINNING, CONTROL_FILE_BEGINNING)
        image = cv2.imread(fileName)

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foundCorners, corners = cv2.findChessboardCorners(grayImage, PATTERN_SIZE, None)
        cornersRefined = cv2.cornerSubPix(grayImage, corners, (11, 11), (-1, -1), CRITERIA)

        reprojectedPoints, _ = cv2.projectPoints(newObjectPoints.astype(np.float32), rotationVecs[imageNumber], translationVecs[imageNumber], cameraMatrix, distCoeffs)
        reprojectedPoints = reprojectedPoints.reshape(-1,2)
        reprojectionError.append(computeMeanReprojectionError(cornersRefined, reprojectedPoints))

        image = cv2.drawChessboardCorners(image, PATTERN_SIZE, reprojectedPoints, foundCorners)
        cv2.imwrite("editedOpenCV_" + fileName, image)

    print("\nControl image OpenCV reprojection error:\n", reprojectionError)
    print("Mean: ", np.mean(reprojectionError))


























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
    M1, X1, t, M2, X2 = affineReconstruction(imagePoints, PATTERN_SIZE)
    endAffine = time.time()
    timeAffine = endAffine - startAffine
    affineCameraMatrizesV1 = np.split(M1, len(images))
    affineCameraMatrizesV2 = np.split(M2, len(images))


    print("\n\nAffine CameraMatrizes V1:")
    print(M1)
    print("\n\nAffine CameraMatrizes V1:")
    print(affineCameraMatrizesV1)
    print("\n3D-Points V1:")
    print(X1)

    print("\n\nAffine CameraMatrizes V2:")
    print(M2)
    print("\n\nAffine CameraMatrizes V2:")
    print(affineCameraMatrizesV2)
    print("\n3D-Points V2:")
    print(X2)
    print("\nt:\n", t)

    affineReprojectedPoints1, affineReprojectionError1 = affineReprojection(affineCameraMatrizesV1, imagePoints, t, X1)
    affineReprojectedPoints2, affineReprojectionError2 = affineReprojection(affineCameraMatrizesV2, imagePoints, t, X2)
    print("\n\n\nAffine Reprojected Points1;\n", affineReprojectedPoints1)
    print("\n\n\nAffine Reprojected Points2;\n", affineReprojectedPoints2)
    print("\nOpenCV Reprojection Errors:\n", meanReprojectionErrors)
    print("Mean: ", np.mean(meanReprojectionErrors))
    print("\n\nAffine Reprojection Error1:\n", affineReprojectionError1)
    print("Mean: ", np.mean(affineReprojectionError1))
    print("\n\nAffine Reprojection Error2:\n", affineReprojectionError2)
    print("Mean: ", np.mean(affineReprojectionError2))
    print("\n\nTime needed by OpenCV: ", timeOpenCV)
    print("Time needed by Affine: ", timeAffine)

    if CONTROLLING:
        movingAffine(X1, affineCameraMatrizesV1, images, t, "")
        movingAffine(X2, affineCameraMatrizesV2, images, t, "2")
        movingCV(objectPoints[0], images, t, rotationVecs, translationVecs, cameraMatrix, distCoeffs)

if __name__ == "__main__":
    main()
