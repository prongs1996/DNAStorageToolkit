import os
import sys
import configparser
import subprocess

BASE_DIR=sys.argv[1]

config_location = sys.argv[2]
config = configparser.ConfigParser()
config.read(config_location)

encode_or_decode  = int(sys.argv[3])  # 0: encode, 1: decode


out_dir=sys.argv[4]

skip_ECC  = int(sys.argv[5])  # 0: encode with RS, 1: encode without RS

parameters = config['parameters']
symbol_size  = int(parameters['symbol_size'])
code_length =  pow(2, symbol_size) -1; 
fec_l=float(parameters['redundancy'])
fec_l = int(fec_l*code_length/100)
file_size = int(parameters['file_size']) #if specified for encoding, uses only the first {file_size} bytes form the file
mapping=int(parameters['mapping_scheme'])

file_locations = config['file_locations']
inputfile=(file_locations['input_f'])
outputfile_encoded=(file_locations['output_f_encoded'])
noisy_strands=(file_locations['noisy_f'])
clustered_strands=(file_locations['clustered_f'])
reconstructed_strands=(file_locations['reconstructed_f'])
outputfile_decoded=(file_locations['output_f_decoded'])
priority_file=(file_locations['priority_f'])


f=open("./constants_RS.hpp","w")
f.write(f'const int REDUNDANCY_SYMBOLS = {fec_l};\n')
f.write(f'const int SYMBOL_SIZE_IN_BITS = {symbol_size};\n')
f.write(f'const int CODE_LENGTH = {code_length};\n')
f.close()


os.system(f'mkdir -p {BASE_DIR}/output/{out_dir}')

if encode_or_decode==0:
	compile_cmd= 'g++ -o3 -fopenmp ./encoder.cpp -o encoder'
	os.system(compile_cmd)
	execute_cmd= f'./encoder {BASE_DIR}/input/{inputfile} {BASE_DIR}/output/{out_dir}/{outputfile_encoded} {file_size} {mapping} {BASE_DIR}/input/priority_maps/{priority_file} {skip_ECC} | cd --'
	os.system(execute_cmd)
else:
	compile_cmd= 'g++ -o3 -fopenmp ./decoder.cpp -o decoder'
	os.system(compile_cmd)
	err_log=open(f'{BASE_DIR}/output/{out_dir}/{outputfile_decoded}.stderr', "w")
	out_log=open(f'{BASE_DIR}/output/{out_dir}/{outputfile_decoded}.stdout', "w")
	subprocess.call(["./decoder", f"{BASE_DIR}/output/{out_dir}/{reconstructed_strands}", f"{BASE_DIR}/output/{out_dir}/{outputfile_decoded}", f"{file_size}", f"{mapping}", f"{BASE_DIR}/input/priority_maps/{priority_file}", f"{skip_ECC}"], stdout=out_log, stderr=err_log)
	err_log.close()
	out_log.close()

