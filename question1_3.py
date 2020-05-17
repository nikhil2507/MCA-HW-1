
import numpy as np
import cv2


# Load the image
img = cv2.imread('test.jpg')

# # Detect the SURF key points
# surf = cv2.xfeatures2d.SURF_create(hessianThreshold=50000, upright=True, extended=True)
# keyPoints, descriptors = surf.detectAndCompute(gray, None)

def Hessian(img,sigma):

    Ixx = ( -1 + (x**2)/(sigma**2) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) / 2*(np.pi)*( sigma**4 ) 
    Iyy = ( -1 + (y**2)/(sigma**2) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) / 2*(np.pi)*( sigma**4 ) 
    Ixy = ( (x*y)/2*(np.pi)*(sigma**6) ) * ( np.exp(-( ((x**2) + (y**2)) / ( sigma**2 )))) 

    # Determinant

    I1 = Ixx * Iyy
    I2 = ((0.9)*Ixy) ** 2

    Det = I1 - I2
        
    return Det

def Detect(det, th):
    keyPoints = []
    for i in range(len(det)):
        # print(1)
        for j in range(len(det[0])):
            # print(1)
            if ( det[i][j] >= th ):
                keyPoints.append(det[i][j])
                # print(keyPoints)
    
    return keyPoints
        
Hes = Hessian(img,1.2)
print(Hes)
points = Detect(Hes, th = 0.001)
print('------------------------------')
# print(Hes[0][0])
print(points)

# Paint the key points over the original image
# result = cv2.drawKeypoints(img, keypoints = points, outImage = None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the results
# cv2.imshow('Key points', result)
cv2.waitKey(0)
cv2.destroyAllWindows()