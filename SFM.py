import cv2
import numpy as np
import glob

images = sorted(glob.glob("frames/*.jpg"))

orb = cv2.ORB_create(3000)

img1 = cv2.imread(images[0])
img2 = cv2.imread(images[1])

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:100],
    None,
    flags=2
)

cv2.imshow("Feature Matches", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
