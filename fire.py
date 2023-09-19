from ultralytics import YOLO
import cvzone
import cv2
import math
import serial

ser = serial.Serial('COM5', 9600)


def drawxy(img, x, y, w, h):
    cv2.line(img, (320, 0), (320, 480), (0, 255, 255), 2)  # dọc  y
    cv2.line(img, (0, 240), (640, 240), (0, 255, 255), 2)  # ngang x

    cv2.putText(img, "x", (620, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "y", (330, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.line(img, (320, y + int(h / 2)), (x + int(w / 2), y + int(h / 2)), (0, 255, 0), 2)  #

    cv2.line(img, (x + int(w / 2), 240), (x + int(w / 2), y + int(h / 2)), (0, 255, 0), 2)  #


def kcy(x, y, w, h):
    # Tọa độ của điểm 1
    # x1, y1 = 0, 240
    # Tọa độ của điểm 2
    # x2, y2 = 320, 240
    # tọa độ điểm 1
    # x1,y1 = 320, y+int(h/2)
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    kc = math.sqrt(((x + int(w / 2) - 320)) ** 2 + ((y + int(h / 2)) - (y + int(h / 2))) ** 2)

    return kc


def kcx(x, y, w, h):
    # Tọa độ của điểm 1
    # x1, y1 = 0, 240
    # Tọa độ của điểm 2
    # x2, y2 = 320, 240
    # tọa độ điểm 1
    # tam_lua = (x+int(w/2), y+int(h/2))

    # x1,y1 = x+int(w/2), 240
    # tọa độ điểm 2
    # x2,y2 = x+int(w/2), y+int(h/2)
    # distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    kc = math.sqrt(((x + int(w / 2)) - (x + int(w / 2))) ** 2 + ((y + int(h / 2)) - 240) ** 2)

    return kc


def main():
    # Running real time from webcam
    cap = cv2.VideoCapture(1)
    model = YOLO('best.pt')
    # Reading the classes
    classnames = ['fire']
    x, y, w, h = 0, 0, 0, 0
    w1, h1 = 0, 0
    # Khởi tạo góc ban đầu của servo X
    angle_x = 90  # Bắt đầu ở 90
    angle_y = 90  # Góc ban đầu của trục thứ hai
    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (640, 480))
        image = cv2.rotate(image, cv2.ROTATE_180)

        result = model(image, stream=True)

        # Getting bbox,confidence and class names informations to work with
        fire_found = False
        for info in result:
            boxes = info.boxes
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 50:
                    x, y, w, h = box.xyxy[0]

                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 5)
                    w1 = w - x
                    h1 = h - y
                    print('tdxywh:', x, y, w, h)
                    cvzone.putTextRect(image, f'{classnames[Class]} {confidence}%', [x + 8, y + 100],
                                       scale=1.5, thickness=2)
                    fire_found = True
                    break
            if fire_found:
                break
        print('kc x:', kcx(x, y, w1, h1))
        print('kc y:', kcy(x, y, w1, h1))
        if kcy(x, y, w1, h1) > 20:
            if x < 320:
                angle_x += 2  # Quay sang phải
            elif x > 320:
                angle_x -= 2  # Quay sang trai
            # góc servo
            angle_x = max(0, min(180, angle_x))
        if kcx(x, y, w1, h1) > 20:
            if (y + h1 / 2) < 240:
                angle_y -= 2  # Quay lên trên
            else:
                angle_y += 2  # Quay xuống
            # góc servo
            angle_y = max(0, min(180, angle_y))
        else:
            angle_x = angle_x
            angle_y = angle_y
        # gửi
        data_to_send = f"{angle_x} {angle_y}\n"
        ser.write(data_to_send.encode())
        drawxy(image, x, y, w1, h1)

        cv2.imshow('image', image)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Close")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
