import cv2

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 20.0)

for i in range(1,20):
    img = cv2.imread('data_{0:d}.png'.format(i))
    video.write(img)

video.release()

