#include <esp_now.h>
#include <WiFi.h>
#include "dataStruct.h"

// Replace with the DAQ's MAC address
uint8_t daq_mac[] = {0x54, 0x43, 0xB2, 0xDC, 0x8C, 0xA0};

com_send_struct send_data;

int current_commanded_state;
short int current_daq_state;

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW initialization failed!");
    return;
  }

  esp_now_peer_info_t peerInfo;
  memcpy(peerInfo.peer_addr, daq_mac, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer!");
    return;
  }

  esp_now_register_send_cb(OnDataSent);
  esp_now_register_recv_cb(OnDataRecv);

  current_commanded_state = 0;
  current_daq_state = 0;
  
}
void SerialRead() {
    if (Serial.available() > 0) {
   current_commanded_state =  Serial.read()-48;
// Serial.print("AVAILABLE--------------------");
   // Serial.println(serialState);
   // Serial.println(" ");
 
  } 
}

void loop() {
  SerialRead();
  // if (Serial.available()) {
  //   String inputString = Serial.readStringUntil('\n');
  //   int inputValue =inputString.toInt();
  //   // Serial.print("Serial Input: ");
  //   // Serial.println(inputValue);
  //   current_commanded_state = inputValue;
  
  // }

// Serial.print("Current command state: ");
    Serial.println(current_daq_state);
if (send_data.commanded_state != current_commanded_state) {
  send_data.commanded_state = current_commanded_state;
  
  esp_err_t result = esp_now_send(daq_mac, (uint8_t *) &send_data, sizeof(send_data));
}
  //   if (result == ESP_OK) {
  //   Serial.println("Data sent!");
  // } else {
  //   Serial.println("Failed to send data!");
  // }


  delay(30);
}

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // if (status == ESP_NOW_SEND_SUCCESS) {
  //   Serial.println("Data was sent successfully!");
  // } else {
  //   Serial.println("Data sent failed!");
  // }
}

void OnDataRecv(const uint8_t *mac_addr, const uint8_t *data, int data_len) {
  com_receive_struct receivedData;
  memcpy(&receivedData, data, sizeof(receivedData));

current_daq_state = receivedData.daq_current_state;
  // Serial.print("daq state: ");
  // Serial.println(receivedData.daq_current_state);

  // // Serial.print("Temperature: ");
  // Serial.println(receivedData.temperature);
  // Serial.print("Humidity: ");
  // Serial.println(receivedData.humidity);
  // Serial.print("Is raining: ");
  // Serial.println(receivedData.isRaining ? "Yes" : "No");
}
