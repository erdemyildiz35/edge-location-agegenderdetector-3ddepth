import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

images = sorted(glob.glob("frames/*.jpg"))

orb = cv2.ORB_create(6000)

# Kamera matrisi
sample = cv2.imread(images[0])
h, w = sample.shape[:2]
focal_length = 800

K = np.array([
    [focal_length, 0, w/2],
    [0, focal_length, h/2],
    [0, 0, 1]
])

all_points = []

for i in range(0, min(20, len(images)-5), 5):

    img1 = cv2.imread(images[i])
    img2 = cv2.imread(images[i+5])

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        continue

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)

    if E is None:
        continue

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    if len(pts1) < 8:
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]

    z = points_3d[2]
    valid = (z > 0) & (z < np.percentile(z, 90))
    points_3d = points_3d[:, valid]

    all_points.append(points_3d)

# Hepsini birleştir
if len(all_points) == 0:
    print("Yeterli 3D nokta üretilemedi.")
    exit()

combined = np.hstack(all_points)

# Normalize
combined = combined / np.max(np.abs(combined))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(combined[0], combined[1], combined[2], s=2)

ax.set_title("Multi-Frame 3D Reconstruction")
plt.show()
