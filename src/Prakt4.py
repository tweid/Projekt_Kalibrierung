import numpy as np
import cv2


#def achtttt_punkttt(p, pp):
#    A =()
#    for i in p.size:


def draw_lines(img1, img2, lines, pts1, pts2):
    r, c, _ = np.shape(img1)
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[0][2] / line[0][1]])
        x1, y1 = map(int, [c, -(line[0][2] + line[0][0] * c) / line[0][1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1[0]), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2[0]), 5, color, -1)
    return img1, img2


def calculate_sum_of_p(pts1, pts2, fund_mat):
    n = 0.0
    count = 0
    for p1, p2 in zip(pts1, pts2):
        line = np.dot(np.append(p1, 1), fund_mat)
        dst = np.dot(line, np.append(p2, 1))
        n += np.abs(dst)
        count += 1
    return n / count


def line_intersect(l1, l2):
    a = np.array(((l1[0][0], l1[0][1]), (l2[0][0], l2[0][1])))
    b = np.array((-l1[0][2], -l2[0][2]))
    x, y = np.linalg.solve(a, b)
    return x, y



def sum_line_intersect(lines):
    x, y = 0.0, 0.0
    count = 0
    for line1 in lines:
        for line2 in lines:
            if not np.array_equal(line1, line2):
                s1, s2 = line_intersect(line1, line2)
                x += s1
                y += s2
                count += 1
    return x / count, y / count


def d_min(line, pt):
    return line[0] * pt[0][0] + line[1] * pt[0][1] + line[2]


def avg_d_min(pts1, pts2, fund_mat):
    sum = 0.0
    count = 0
    for pt1, pt2 in zip(pts1, pts2):
        line1 = np.dot(fund_mat, np.append(pt1, 1))
        line2 = np.dot(fund_mat, np.append(pt2, 1))
        sum += np.abs(d_min(line1, pt2))
        sum += np.abs(d_min(line2, pt1))
        count += 1
    return sum / (2 * count)


def main():
    print("test")

    patternSize = (9, 6)


    imgLeft = cv2.imread("TI119_L.jpg")
    imgRight = cv2.imread("TI119_R.jpg")

    leftCorners = cv2.findChessboardCorners(imgLeft, patternSize)[1]
    rightCorners = cv2.findChessboardCorners(imgRight, patternSize)[1]

    print("left corners:")
    print(leftCorners)
    print("")
    print("rigth corners:")
    print(rightCorners)

    print("")
    print("")
    print("")

    fundamental = cv2.findFundamentalMat(leftCorners, rightCorners)[0]

    leftLines = cv2.computeCorrespondEpilines(leftCorners, 1, fundamental)
    rightLines = cv2.computeCorrespondEpilines(rightCorners, 2, fundamental)

    print("Fundamental:")
    print(fundamental)
    print()
    print()
    #print("Left Epiline")
    #print(leftLines)
    #print()
    #print("Right Epiline")
    #print(rightLines)

    print()
    print()
    print()
    print()
    print()


    draw_lines(imgLeft, imgRight, rightLines, leftCorners, rightCorners)

    cv2.imwrite("_editedLeft.jpg", imgLeft)
    cv2.imwrite("_editedRight.jpg", imgRight)

    determin = np.linalg.det(fundamental)
    print("Determinante")
    print(determin)
    print()
    print()
    print()
    print()
    print()

    summe = calculate_sum_of_p(leftCorners, rightCorners, fundamental)
    print("Summe:")
    print(summe)
    print()
    print()
    print()
    print()
    print()



    rightEpipole = sum_line_intersect(rightLines)
    leftEpipole = sum_line_intersect(leftLines)
    print("Rechter Epipol:")
    print(rightEpipole)
    print()
    print()
    print()
    print()
    print()
    print("Linker Epipol:")
    print(leftEpipole)
    print()
    print()
    print()
    print()

    #cv2.imshow("original Links", imgLeft)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #Get Matrix A
    #n = anzahl der Punkte
    #Long[][] A = new Long[n][n];
    #for i < n
    #   A[n][i] = 1;
    #for



#     img = cv2.imread('BinaryObjectsI.png', 0)
#
# out = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
# cv2.normalize(out, out, 0, 1, cv2.NORM_MINMAX)
#
# maxima_gray = out.copy()
# maxima = cv2.cvtColor(maxima_gray, cv2.COLOR_GRAY2BGR)
# local_maxima(out, maxima)
#
# cv2.imshow('transformed', maxima)
# cv2.imshow('original', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



#     Mp = [[-1115.7682, 923.97387, 679.2890, 127500],
#           [850.8388, 31.08950, 1354.8917, -222750],
#           [0.4933, 0.81276, -0.3100, 50]]
#
#     pixel_per_mm = 400
#
#     zeroes = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
#
#     T = [[-1115.7682, 923.97387, 679.2890],
#          [850.8388, 31.08950, 1354.8917],
#          [0.4933, 0.81276, -0.3100]]
#
#     r = np.zeros((3, 3), dtype = "float32")
#     q = np.zeros((3, 3), dtype = "float32")
#     src = np.array(T, dtype = "float32")
#     cv2.RQDecomp3x3(src, mtxR = r, mtxQ = q);
#
#     Mint = r
#     R = q
#
#     a_x = Mint[0][0]
#     brennweite = a_x / pixel_per_mm
#
#     print("M_int")
#     print(Mint)
#     print("R")
#     print(R)
#     print("")
#     print("Brennweite")
#     print(brennweite)
#
#     inverse = np.linalg.inv(Mint)
#
#     Mext = np.matmul(inverse, Mp)
#     t = [Mext[0][3], Mext[1][3], Mext[2][3]]
#
#     print("t")
#     print(t)
#
#
#
#     beta = np.arcsin(Mext[0][2])
#     gamma = np.arccos(Mext[0][0] / np.cos(beta))
#     alpha = np.arccos(Mext[2][2] / np.cos(beta))
#
#     print()
#     print("alpha")
#     print(alpha * 180 / np.pi)
#     print("beta")
#     print(beta * 180 / np.pi)
#     print("gamma")
#     print(gamma * 180 / np.pi)





if __name__ == "__main__":
    main()
