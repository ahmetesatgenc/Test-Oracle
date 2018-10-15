import cv2

def BFmatcher(des1, des2):
    bf = cv2.BFMatcher()
    matches12 = bf.knnMatch(des1, des2, k=2)
    matches21 = bf.knnMatch(des2, des1, k=2)
    goodMatches12 = [m for (m, n) in matches12 if m.distance < 0.7 * n.distance]
    goodMatches21 = [m for (m, n) in matches21 if m.distance < 0.7 * n.distance]
    goodMatches = []
    if goodMatches12 is not None and goodMatches21 is not None:
        for m1 in goodMatches12:
            for m2 in goodMatches21:
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    goodMatches.append(m1)
                    break
    else:
        goodMatches.append(0)

    return matches12, goodMatches