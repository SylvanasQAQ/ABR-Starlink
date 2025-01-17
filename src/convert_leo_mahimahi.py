import numpy as np
import os

BYTES_PER_PKT = 1500  
BITS_PER_BYTE = 8
MILLISEC_IN_SEC = 1000.0

def convert_to_mahimahi_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        next(infile) 
        for line in infile:
            time, speed, rtt = line.strip().split(',')
            time = float(time) * 1000 
            speed = float(speed) * 1e6 
            rtt = float(rtt) 

            pkt_per_millisec = speed / (BYTES_PER_PKT * BITS_PER_BYTE) / MILLISEC_IN_SEC
            millisec_count = 0
            millisec_time = int(time)
            pkt_count = 0

            while millisec_count < 1000: 
                to_send = (millisec_count * pkt_per_millisec) - pkt_count
                to_send = np.floor(to_send)
                for _ in range(int(to_send)):
                    outfile.write(str(millisec_time) + '\n')
                pkt_count += to_send
                millisec_count += 1
                millisec_time += 1


input_folder = 'test'
output_folder = 'mahimahi_traces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith('.log'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename.replace('.log', '.mahimahi'))
        convert_to_mahimahi_format(input_file, output_file)