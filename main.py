import cv2
import numpy as np
import math

def redcolor(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower = [0,100,100]
    upper = [10,255,255]
    
    lower_red = np.array(lower)  # Giá trị thấp của màu đỏ
    upper_red = np.array(upper)  # Giá trị cao của màu đỏ

    # Tạo mask dựa trên phạm vi màu đỏ
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Áp dụng mask lên ảnh gốc
    result = cv2.bitwise_and(image, image, mask=mask)

    # Hiển thị ảnh gốc và ảnh sau khi nhận diện màu đỏ
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Red Detection', result) 
def detecfire(image):
    
    blur = cv2.GaussianBlur(image, (15,15), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    cv2.imshow('HSV', hsv)
    
    lower = [18,50,50]
    upper = [35,255,255]
    
    lower = np.array(lower, dtype= 'uint8')
    upper = np.array(upper, dtype= 'uint8')
    
    mask = cv2.inRange(hsv, lower, upper)
    
    cv2.imshow('mask', mask)
    
    img_out = cv2.bitwise_and(image, hsv, mask= mask)
    
    cv2.imshow('Original Image', image)
    cv2.imshow('FIRE', img_out)
    
    return img_out


def drawxy(img,x,y,w,h):
    cv2.line(img, (320,0), (320, 480), (0,255,255), 2) # dọc  y
    cv2.line(img, (0,240), (640, 240), (0,255,255), 2) # ngang x
    
    cv2.putText(img, "x" , (620 , 230), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)
    cv2.putText(img, "y" , (330 , 20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2, cv2.LINE_AA)
    
    
    cv2.line(img, (320,y+int(h/2)), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
    cv2.line(img, (x+int(w/2), 240), (x+int(w/2), y+int(h/2)), (0,255,0), 2) # 
def distances(x,y,w,h):
    tam_lua = (x+int(w/2), y+int(h/2))
    # Tọa độ của điểm 1
    x1, y1 = 0, 240
    # Tọa độ của điểm 2
    x2, y2 = 320, 240
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # In ra khoảng cách
    print(f'kc: {distance}')   
    print(tam_lua) 

def main():
    fire_cascade = cv2.CascadeClassifier('fire_detection.xml')
    cam = cv2.VideoCapture(1)
    
    x,y,w,h = 0,0,0,0
    while True:
        _, image = cam.read()   
        redcolor(image)
        #image = cv2.resize(image_raw, (1024, 765))
        fire = fire_cascade.detectMultiScale(image, 1.2, 5)
        
        for (x,y,w,h) in fire:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            
        distances(x,y,w,h) 
        drawxy(image,x,y,w,h)
        
        cv2.imshow('fr', image)
        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Close")
            break
    cam.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
        
#edit test
    