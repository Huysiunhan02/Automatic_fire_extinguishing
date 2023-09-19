#include <Servo.h>
Servo servoX;
Servo servoY;
void setup() {
  servoX.attach(D2);  
  servoY.attach(D3);  
  Serial.begin(9600);
  servoX.write(90);  
  servoY.write(90);  
}
void loop() {
  if (Serial.available() > 0) {
    int angleX = Serial.parseInt();
    int angleY = Serial.parseInt();
    servoX.write(angleX);
    servoY.write(angleY);
    delay(100);  
  }
}
