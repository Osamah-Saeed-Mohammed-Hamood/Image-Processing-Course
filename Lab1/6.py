import cv2 as c

videoCapture = c.VideoCapture('images/Vedio.mp4')

if not videoCapture.isOpened():
    print("Error: Could not open vedio file .")
    exit()

fps = videoCapture.get(c.CAP_PROP_FPS)
size = (int(videoCapture.get(c.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(c.CAP_PROP_FRAME_HEIGHT)))

videoWriter = c.VideoWriter('images/3.mp4',c.VideoWriter_fourcc(*'mp4v'),fps,size)

success ,frame = videoCapture.read()
while success:
    videoWriter.write(frame)
    c.imshow('Video Preview',frame)
    if c.waitKey(1)==ord('q'):
        break
    success , frame = videoCapture.read()

videoCapture.release()
videoWriter.release()
c.destroyAllWindows()
print("Video Copying completed successfully")