import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

#kameradan görüntü alabilmemiz için bu fonksiyonu kullanıyoruz..
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        #print(face_loc) -> aldığımız 4 değerler y1,x1,y2,x2i eşleştiryotuz.
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 238), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 238), 4)

    cv2.imshow("Yuz Tanima Ekrani", frame)

    #q tuşuna basıldığı zaman program dursun.
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()