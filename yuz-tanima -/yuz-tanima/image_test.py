import cv2
import face_recognition

image = cv2.imread("aylin.jfif")
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

image2 = cv2.imread("images/aylin.png")
rgb_img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv2.imshow("Photo", image)
cv2.imshow("Test Photo", image2)
cv2.waitKey(0)