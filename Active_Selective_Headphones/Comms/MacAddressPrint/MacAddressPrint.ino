#include <WiFi.h>

void setup() {
  Serial.begin(115200);

  // Set the ESP32 as a station (client)
  WiFi.mode(WIFI_STA);
  
  // Get the MAC address
  uint8_t mac[6];
  WiFi.macAddress(mac);
  
  // Print the MAC address
  Serial.print("ESP32 MAC Address: ");
  for (int i = 0; i < 6; i++) {
    if (mac[i] < 16) {
      Serial.print("0");
    }
    Serial.print(mac[i], HEX);
    if (i < 5) {
      Serial.print(":");
    }
  }
  Serial.println();
}

void loop() {
}

