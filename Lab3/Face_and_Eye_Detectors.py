import cv2 as cv

face = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
eye = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_eye.xml')

img = cv.imread("images/R.jpeg")
img = cv.resize(img,(500,500))
img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

faces = face.detectMultiScale(img_gray,1.3,5)

for (x,y,w,h) in faces:
    img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = img_gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye.detectMultiScale(roi_gray,1.3,5)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)
cv.imshow('Image 1',img)
cv.waitKey(0)
cv.destroyAllWindows()
