#ifndef DATASTRUCT_H
#define DATASTRUCT_H

// DAQ receive struct
typedef struct daq_receive_struct {
    short int commanded_state;
} daq_receive_struct;

// COM receive struct
typedef struct daq_send_struct {
    short int daq_current_state;
    // Add more fields as needed for the receiver
} daq_send_struct;

#endif
