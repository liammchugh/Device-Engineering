#include <WiFi.h>
#include "wifi_config.h"

const char* ssid = SSID;
const char* password = PASSWORD;
const char* serverIP = IP;
const uint16_t serverPort = 80;

double i = 0.0;
// double waitTime = 50;

WiFiClient client;

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");

  if (!client.connect(serverIP, serverPort)) {
    Serial.println("Connection to server failed");
    return;
  }

  Serial.println("Connected to server");
}

void loop() {
  if (client.connected()) {
    // Check if data is available from the server
    if (client.available()) {
      String received = client.readStringUntil('\n');
      double receivedDouble = received.toDouble();
      Serial.print("Received double: ");
      Serial.println(receivedDouble, 6); // Print the received double with 6 decimal places
    }

    // Calculate the next value to send
    double y = sin(i);
    i = i + waitTime * pow(10, -3);
    sendDouble(y);

    
  } else {
    Serial.println("Disconnected from server");
    delay(5000);

    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Reconnection failed");
    } else {
      Serial.println("Reconnected to server");
    }
  }
  delay(waitTime);
}

void sendDouble(double value) {
  String valueStr = String(value, 6); // Convert the double value to a string with 6 decimal places
  client.print(valueStr);
  client.print('\n'); // Send a newline character to indicate the end of the value
  client.flush();
}

