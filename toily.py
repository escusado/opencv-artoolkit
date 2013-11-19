# import numpy as np
# import cv2

# # Load an color image in grayscale
# img = cv2.imread('at.jpg',0)


# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

################################################################################

# import numpy as np
# import cv2

# cap = cv2.VideoCapture('./testFiles/video.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     frameImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',frameImage)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break



# cap.release()
# cv2.destroyAllWindows()

################################################################################

import cv2
import numpy as np

def find_corners(image):
    im = cv2.Canny(image, 100, 200)

    cnt = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = cv2.approxPolyDP(cnt[0], 5, True)
    return cnt.astype(np.float32)

def main():
    trap = cv2.imread('trap2.jpg', 0)
    rect = cv2.imread('rect.jpg', 0)
    print('>>>', trap)

    ptsTrap = find_corners(trap)
    ptsRect = find_corners(rect)

    T = cv2.getPerspectiveTransform(ptsTrap, ptsRect)

    warp = cv2.warpPerspective(trap, T, rect.shape[:2])

    cv2.imshow('', warp)
    cv2.imwrite('warp.png', warp)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

################################################################################


# import numpy as np
# import cv2

# im = cv2.imread('./testFiles/treapezoid.jpg')
# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
