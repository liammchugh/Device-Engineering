#include <WiFi.h>
#include "wifi_config.h"

const char* ssid = SSID;
const char* password = PASSWORD;
WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  server.begin();
}

void loop() {
  WiFiClient client = server.available();

  if (client) {
    Serial.println("Client connected");
    while (client.connected()) {
      double receivedDouble = 0;
      // Check if data is available from the client
      if (client.available()) {
        String received = client.readStringUntil('\n');
      receivedDouble = received.toDouble();
        // Serial.print("Received double: ");
        Serial.println(receivedDouble, 6); // Print the received double with 6 decimal places

       
      }

       // Echo the received double value back to the client
        // String valueStr = String(receivedDouble, 6);
        // client.print(valueStr);
        // client.print('\n');
        // client.flush();
      delay(50);
    }
    client.stop();
    Serial.println("Client disconnected");
  }
}
