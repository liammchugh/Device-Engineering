#include <LinxESP32.h>




void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

}
double i = 0.0;
double waitTime = 10;
void loop() {
  // put your main code here, to run repeatedly:

  double y = sin(i);


  Serial.println(y);
  i = i+waitTime*pow(10, -3);
  
  delay(waitTime);

}
