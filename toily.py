import cv2
import numpy as np

def find_corners(image):
    im = cv2.Canny(image, 100, 200)

    cnt = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = cv2.approxPolyDP(cnt[0], 5, True)
    return cnt.astype(np.float32)

def main():
    trap = cv2.imread('trap.jpg', 0)
    rect = cv2.imread('rect.jpg', 0)
    sasha = cv2.imread('sasha.jpg', 0)


    ptsTrap = find_corners(trap)
    ptsRect = find_corners(rect)

    T = cv2.getPerspectiveTransform(ptsRect, ptsTrap)

    warp = cv2.warpPerspective(sasha, T, rect.shape[:2])

    cv2.imshow('', warp)
    cv2.imwrite('warp.png', warp)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()