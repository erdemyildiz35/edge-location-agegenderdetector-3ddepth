import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

images = sorted(glob.glob("frames/*.jpg"))

orb = cv2.ORB_create(5000)

img1 = cv2.imread(images[0])
img2 = cv2.imread(images[8])  # biraz daha uzak frame

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Lowe ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

# Camera intrinsics (yaklaşık)
focal_length = 800
h, w = gray1.shape
K = np.array([
    [focal_length, 0, w/2],
    [0, focal_length, h/2],
    [0, 0, 1]
])

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)

if E is None or mask is None:
    print("Essential matrix didnt calculated ")
    exit()

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

print("Inlier :", len(pts1))


_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K @ np.hstack((R, t))

points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
points_3d = points_4d[:3] / points_4d[3]

# Aşırı uzak noktaları temizle
z = points_3d[2]
valid = (z > 0) & (z < np.percentile(z, 90))

points_3d = points_3d[:, valid]

# Normalize et
points_3d = points_3d / np.max(np.abs(points_3d))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[0], points_3d[1], points_3d[2], s=3)

ax.set_title("Cleaned 3D Reconstruction")
plt.show()
