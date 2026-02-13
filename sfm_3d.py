import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

images = sorted(glob.glob("frames/*.jpg"))

orb = cv2.ORB_create(4000)

img1 = cv2.imread(images[0])
img2 = cv2.imread(images[5])

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# Kamera matrisi (yaklaşık değer)
focal_length = 800
h, w = gray1.shape
K = np.array([
    [focal_length, 0, w/2],
    [0, focal_length, h/2],
    [0, 0, 1]
])

# Essential Matrix
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Pose recovery
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

# Projection matrices
P1 = np.hstack((np.eye(3), np.zeros((3,1))))
P2 = np.hstack((R, t))

P1 = K @ P1
P2 = K @ P2

# Triangulation
points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

points_3d = points_4d[:3] / points_4d[3]

# Plot 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[0], points_3d[1], points_3d[2], s=1)

plt.show()
