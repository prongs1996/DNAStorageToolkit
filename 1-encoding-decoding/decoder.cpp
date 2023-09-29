#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <inttypes.h>
#include <iomanip>

#define NO_GFLUT

#include "schifra_galois_field.hpp"
#include "constants_RS.hpp"

#undef NO_GFLUT

#include "schifra_galois_field_polynomial.hpp"
#include "schifra_sequential_root_generator_polynomial_creator.hpp"
#include "schifra_reed_solomon_encoder.hpp"
#include "schifra_reed_solomon_decoder.hpp"
#include "schifra_reed_solomon_block.hpp"
#include "schifra_error_processes.hpp"


#define RS_ROWS 14

/* Finite Field Parameters */
const std::size_t field_descriptor                =   SYMBOL_SIZE_IN_BITS;
const std::size_t generator_polynomial_index      =   0;
const std::size_t generator_polynomial_root_count =  REDUNDANCY_SYMBOLS;
/* Reed Solomon Code Parameters */
const std::size_t code_length = CODE_LENGTH;
const std::size_t fec_length  = REDUNDANCY_SYMBOLS;
const std::size_t data_length = code_length - fec_length;



int DNAtoOrd(std::string strand){
  int res=0;
  for(int i=0; i<8; i++){
     res=res<<2;	
     switch(strand.at(i))
     {
 	case 'A': res+=0; break; 
 	case 'C': res+=1; break; 
 	case 'G': res+=2; break; 
 	case 'T': res+=3; break; 
     }
   	
  }
 return res;
}


unsigned char DNAtoC1(std::string strand, int row){
  unsigned char res=0;
  for(int i=0; i<4; i++){
     res= res<<2;
     switch(strand.at(8+8*row+i))
     {
        case 'A': res+=0; break;
        case 'C': res+=1; break;
        case 'G': res+=2; break;
        case 'T': res+=3; break;
     }

  }
 return res;
}


unsigned char DNAtoC2(std::string strand, int row){
  unsigned char res=0;
  for(int i=0; i<4; i++){
     res= res<<2;
     switch(strand.at(8+8*row+i+4))
     { 
        case 'A': res+=0; break;
        case 'C': res+=1; break;
        case 'G': res+=2; break;
        case 'T': res+=3; break;
     }
         
  }
 return res;
}

uint8_t placeBitAccToPriority(int positionInChar, /*comes from the priority array(where the bit needs to be placed)*/ 
			      int bitNumber,      /*The position of bit to be read from the decoded matrix(char)*/
			      char* inputstring) 
{
    int charNum = bitNumber/8;
    int rem = bitNumber%8;
    uint8_t temp =static_cast<uint8_t> (inputstring[charNum]);
    uint8_t mask = static_cast<uint8_t>(1) << (7-rem); 
    temp = ((temp &  mask) << (rem)) >> positionInChar;
    return temp;
}


void WriteRSMatrixByPriority(int inputSize, char *priority_file, char decoded_matrix[][2*data_length],char *output_file)
{
    std::ifstream in_stream(priority_file);
    if (!in_stream){
        std::cout << "Error: priority file could not be opened." << std::endl;
        return;
    }

    int priority_array_size = inputSize*8;
    int *priority_array = new int[priority_array_size];
    std::string prio;
    int i=0; 
    while(std::getline(in_stream, prio) && i<priority_array_size) 
        priority_array[i++]=atoi(prio.c_str());
    in_stream.close();
    assert(i==priority_array_size);
   
    //Linearize the decoded data
    char *sortedarray= new char[inputSize]; 
    int elementcounter=0;
    int top_row=0;
    int bottom_row=RS_ROWS-1;
    int rows_to_read = (inputSize/(data_length*2))+1;
    if(rows_to_read>RS_ROWS) rows_to_read = RS_ROWS;
    int row;
    
    for(int i=0; i< rows_to_read; i++){
        if(i%2==1)  row=top_row++;
        else        row=bottom_row--;

        for(int j=0; j<data_length*2; j++){
            if(elementcounter==inputSize) break;
            elementcounter++;
            sortedarray[(i*data_length*2)+j]=decoded_matrix[row][j];
        }
    }

    char *output_array= new char[inputSize];
    //Unpack the sorted array using priority to get the original input
    for(int i=0; i< priority_array_size;i++){
        int charnum= priority_array[i]/8;
        int rem = priority_array[i]%8;
        uint8_t outputchar = placeBitAccToPriority(rem, i, sortedarray);
        output_array[charnum]= static_cast<char>(static_cast<uint8_t>(output_array[charnum]) | outputchar);
    }

    //Write array to a file
    std::ofstream out_stream(output_file, std::ios::binary);
    if (!out_stream) {
        std::cout << "Error: output file could not be created." << std::endl;
        return ;
    }
    for(int i=0; i<inputSize; i++){
        out_stream.write(&output_array[i],1);
    }
    out_stream.close();
    delete [] output_array;
    delete [] sortedarray;
}


void WriteRSMatrixByColumn(int inputSize, char decoded_matrix[][data_length*2],char* output_file){

    std::ofstream out_stream(output_file, std::ios::binary);
    if (!out_stream) {
        std::cout << "Error: output file could not be created." << std::endl;
        return ;
    }

    for(int i=0; i<inputSize/2; i++)
    {
        int row    = i % RS_ROWS;
        int column = i / RS_ROWS;
        out_stream.write(&decoded_matrix[row][2*column], 2);
    }

    if(inputSize%2!=0){
        int ind    = inputSize/2;
        int row    = ind % RS_ROWS;
        int column = ind / RS_ROWS;
        out_stream.write(&decoded_matrix[row][2*column], 1);
    }

    out_stream.close();
}

int main(int argc, char *argv[])
{

    int inputSize=atoi(argv[3]);
    int total_symbols = RS_ROWS*code_length;
    int total_bytes   = 2*total_symbols;

    unsigned char input_bytes[RS_ROWS][2*code_length];
    std::memset(&input_bytes[0], 0, total_bytes);

    std::ifstream in_stream(argv[1]);
    if (!in_stream) {
        std::cout << "Error: input file could not be opened." << std::endl;
        return 1;
    }
    
    int erasures[code_length]={0};
    int erasure_count=code_length;

    std::string strand; 
    while(std::getline(in_stream, strand)) {
        int column=DNAtoOrd(strand);
        erasures[column]=1;
	erasure_count--;
        for(int row=0; row<RS_ROWS; row++){
            input_bytes[row][2*column]=DNAtoC1(strand, row);  	
            input_bytes[row][2*column+1]=DNAtoC2(strand, row); 
        } 	
    }
    in_stream.close();

    schifra::reed_solomon::erasure_locations_t erasure_location_list;
    
    for(int i=0; i< code_length; i++)
    {
        if(erasures[i] == 0) 
            erasure_location_list.push_back(i);
    }

    uint16_t RS_matrix[RS_ROWS][code_length];
    std::memset(&RS_matrix[0][0], 0, total_bytes);

    for(int i=0; i<RS_ROWS; i++)
      for(int j=0; j<code_length; j++)
         RS_matrix[i][j]= (input_bytes[i][2*j] << 8)  + input_bytes[i][2*j+1];


    /* Instantiate Finite Field and Generator Polynomials */
    const schifra::galois::field field(field_descriptor,
                                      schifra::galois::primitive_polynomial_size14,
                                      schifra::galois::primitive_polynomial14);

    schifra::galois::field_polynomial generator_polynomial(field);

    if  (
        !schifra::make_sequential_root_generator_polynomial(field,
                                                            generator_polynomial_index,
                                                            generator_polynomial_root_count,
                                                            generator_polynomial)
        )
    {
        std::cout << "Error - Failed to create sequential root generator!" << std::endl;
         return 1;
    }

    typedef schifra::reed_solomon::decoder<code_length,fec_length,data_length> decoder_t;
    const decoder_t RS_decoder(field, generator_polynomial_index);

    char decoded_matrix[RS_ROWS][2*data_length];
    std::memset(&decoded_matrix[0][0], 0, 2*RS_ROWS*data_length);

    int decoded[RS_ROWS];
    int detected[RS_ROWS];
    int corrected[RS_ROWS];

    int skip_ECC=atoi(argv[6]);

    if(!skip_ECC){
      #pragma omp parallel for
      for(int i=0; i<RS_ROWS; i++)
      {
	  detected[i]=corrected[i]=0;
	  decoded[i]=1;
         /* Instantiate RS Block For Codec */
         schifra::reed_solomon::block<code_length,fec_length> block;
         schifra::reed_solomon::copy(&RS_matrix[i][0], code_length, block);

         if (!RS_decoder.decode(block,erasure_location_list))
       	  {
	    decoded[i]=0;
            std::cout << "Row "<<i<<": Critical decoding failure, message= " << block.error_as_string() << std::endl;
          }
	  detected[i]=block.errors_corrected;
  	  corrected[i]=block.errors_corrected;
          for(int j=0; j< data_length; j++){
             uint16_t temp = static_cast<uint16_t>(block.data[j]);
             decoded_matrix[i][2*j] = (temp >> 8) & 0xFF;
             decoded_matrix[i][2*j+1] = temp & 0xFF;
          }
      }

      //for incorrectly decoded rows, copy the data before correction 	
      for(int i=0; i<RS_ROWS; i++)
   	  if(decoded[i]==0)     
            for(int j=0; j< data_length; j++)
	    {
             uint16_t temp=RS_matrix[i][j];	   
             decoded_matrix[i][2*j] = (temp >> 8) & 0xFF;
             decoded_matrix[i][2*j+1] = temp & 0xFF;
	    }



    }

    else{
      for(int i=0; i<RS_ROWS; i++)
       {
         schifra::reed_solomon::block<code_length,fec_length> block;
         schifra::reed_solomon::copy(&RS_matrix[i][0], code_length, block);
         for(int j=0; j< data_length; j++){
             uint16_t temp = static_cast<uint16_t>(block.data[j]);
             decoded_matrix[i][2*j] = (temp >> 8) & 0xFF;
             decoded_matrix[i][2*j+1] = temp & 0xFF;
          }
	}
    }


    std::cout<<"-----------------------------------------------------------"<<std::endl;
    std::cout<<"Row number |  decoded? | errors detected | errors corrected " <<std::endl;
    std::cout<<"-----------------------------------------------------------"<<std::endl;

    for(int i=0; i<RS_ROWS; i++)
        std::cout<<std::setw(7)<<i<<"    |" <<std::setw(6)<< decoded[i] <<"     |"<<std::setw(10)<<detected[i]<<"       |"<<std::setw(10)<<corrected[i]<<std::endl;

    std::cout<<"-----------------------------------------------------------"<<std::endl;
    std::cout<<"\nErasures in each row: "<< erasure_count<<std::endl;
    std::cout<<"-----------------------------------------------------------"<<std::endl;

    //if statement to either decode by column or priority
    int mapping = atoi(argv[4]);
    if(mapping==0) WriteRSMatrixByColumn(inputSize, decoded_matrix, argv[2]);
    else WriteRSMatrixByPriority(inputSize, argv[5], decoded_matrix, argv[2]);

    return 0;
}
