#include <sys/stat.h>
#include<cstdlib>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <inttypes.h>
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
#include <bits/stdc++.h> 

#define RS_ROWS  14 


/* Finite Field Parameters */
const std::size_t field_descriptor                =   SYMBOL_SIZE_IN_BITS;
const std::size_t generator_polynomial_index      =  0;
const std::size_t generator_polynomial_root_count =  REDUNDANCY_SYMBOLS;
/* Reed Solomon Code Parameters */
const std::size_t code_length = CODE_LENGTH;
const std::size_t fec_length  = REDUNDANCY_SYMBOLS;
const std::size_t data_length = code_length - fec_length;


const char DNA[4] = {'A', 'C', 'G', 'T'};

std::string ordToDNA(uint16_t ord) //16 bits
{
  std::string s="";
  for(int i=0; i<8; i++){
      s+=DNA[ord%4];
      ord=ord>>2;
  }
  reverse(s.begin(), s.end());
  return s; 
}

std::string symbolToDNA(unsigned char c1, unsigned char c2)
{
  std::string s="";
  for(int i=0; i<4; i++){
      s+=DNA[c2%4];
      c2=c2/4;
  }
  for(int i=0; i<4; i++){
      s+=DNA[c1%4];
      c1=c1/4;
  }
  reverse(s.begin(), s.end());
  return s; 
}


/* extracts $bitNumber-th bit from an input character string and 
 * returns a 16-bit symbol with that bit in position $positioInSymbol (rest is 0) 
 * */
uint16_t getBitByImportance(int positionInSymbol, int bitNumber, char* inputstring) 
{
    int charNum = bitNumber/8;
    int rem = bitNumber%8;
    uint16_t temp = static_cast<uint16_t>(inputstring[charNum]);
    uint16_t mask = static_cast<uint16_t>(1) << (7-rem); 
    /*
    switch(rem){
        case 0:
            mask = 0x0080;
            break;
        case 1:
            mask = 0x0040;
            break;
        case 2:
            mask = 0x0020;
            break;
        case 3:
            mask = 0x0010;
            break;
        case 4:
            mask = 0x0008;
            break;
        case 5:
            mask = 0x0004;
            break;
        case 6:
            mask = 0x0002;
            break;
        case 7:
            mask = 0x0001;
            break;
    }*/

    temp = ((temp &  mask)<<(8+rem)) >> positionInSymbol;
    return temp;
}

void fillRSMatrixByColumn(char *inputBytes, int inputSize,  uint16_t RS_Matrix[][code_length])
{
    for(int i=0; i<inputSize/2; i++)
    {
    int row    = i % RS_ROWS;
        int column = i / RS_ROWS;
        RS_Matrix[row][column]= (static_cast<unsigned char> (inputBytes[2*i]) << 8)  + static_cast<unsigned char> (inputBytes[2*i+1]);
    }

    if(inputSize%2 != 0){
        int ind    = inputSize/2;
        int row    = ind % RS_ROWS;
        int column = ind / RS_ROWS;
        RS_Matrix[row][column]= static_cast<unsigned char>(inputBytes[2*ind]) << 8;
    }
}

void fillRSMatrixByPriority(char *inputBytes, int inputSize,  char *priority_file,  uint16_t RS_Matrix[][code_length])
{

        std::ifstream in_stream(priority_file);
        if (!in_stream) {
            std::cout << "Error: priority file could not be opened." << std::endl;
            return;
        }
        int priority_array_size=inputSize*8;
        int *priority_array =new int[priority_array_size];
        std::string prio;
        int i=0;
        while(std::getline(in_stream, prio) && i<priority_array_size) 
            priority_array[i++]=atoi(prio.c_str());    
        in_stream.close();
        assert(i==priority_array_size);
    

        int sortedarray_size=(inputSize+1)/2;
        uint16_t *sorted_input = new uint16_t[sortedarray_size];

        //Reordering the data
        for(int i=0; i<(inputSize/2); i++)
         {
            uint16_t inputsymbol=0;
            uint16_t symbol;
            for(int j=0; j<16; j++)
             {
                symbol = getBitByImportance(j, priority_array[i*16+j], inputBytes);
                inputsymbol= inputsymbol | symbol;
             }
             sorted_input[i]=inputsymbol;
         }
       if(inputSize%2==1)
        {
            uint16_t inputsymbol=0;
            uint16_t symbol;
            for(int j=0; j<8; j++)
            {
                symbol = getBitByImportance(j, priority_array[(inputSize/2)*16 + j], inputBytes);
                inputsymbol= inputsymbol | symbol;  
            }
            sorted_input[sortedarray_size]=inputsymbol;
        }
   
        //Putting the sorted data into RS_Matrix is alternate row major format
        int elementcounter=0;
        int top_row =0;
        int bottom_row= RS_ROWS-1;
        int rows_to_fill = (sortedarray_size +data_length-1)/data_length;
        if(rows_to_fill>RS_ROWS) rows_to_fill = RS_ROWS; 
        int row;

        for(int i=0; i< rows_to_fill; i++)
         {
            if(i%2==1) row=top_row++;
            else       row=bottom_row--;

            for(int j=0; j<data_length;++j)
            {
                if(elementcounter==sortedarray_size) break;
                RS_Matrix[row][j]= sorted_input[elementcounter];
                elementcounter++;
            }

         }

        delete [] priority_array;
        delete [] sorted_input;
    
}


int main(int argc, char** argv)
{
    std::ifstream in_stream(argv[1], std::ios::binary);
    if (!in_stream) {
        std::cout << "Error: input file could not be opened." << std::endl;
        return 1;
    }

    int inputsize=atoi(argv[3]);
    int total_symbols = RS_ROWS*code_length;
    int total_bytes   = 2*total_symbols;

    char* input_bytes = new char[inputsize];
    std::memset(&input_bytes[0], 0, inputsize);
    in_stream.read(&input_bytes[0], static_cast<std::streamsize>(inputsize));
    in_stream.close();
   
    int mapping=atoi(argv[4]);

    int skip_ECC=atoi(argv[6]);
   
    
    uint16_t RS_matrix[RS_ROWS][code_length];
    std::memset(&RS_matrix[0][0], 0, total_bytes);

    if(mapping==0) fillRSMatrixByColumn(input_bytes, inputsize, RS_matrix);
    else fillRSMatrixByPriority(input_bytes, inputsize,  argv[5],  RS_matrix);
    
    delete [] input_bytes;

    /* Instantiate Finite Field and Generator Polynomials */

    const schifra::galois::field field(field_descriptor,
                                      schifra::galois::primitive_polynomial_size14,
                                      schifra::galois::primitive_polynomial14);

    schifra::galois::field_polynomial generator_polynomial(field);

    if (
            !schifra::make_sequential_root_generator_polynomial(field,
                                                            generator_polynomial_index,
                                                            generator_polynomial_root_count,
                                                            generator_polynomial)
      )
    {
        std::cout << "Error - Failed to create sequential root generator!" << std::endl;
        return 1;
    }

    /* Instantiate Encoder and Decoder (Codec) */
    typedef schifra::reed_solomon::encoder<code_length,fec_length,data_length> encoder_t;
    const encoder_t encoder(field, generator_polynomial);
  
    char output_matrix[RS_ROWS][2*code_length];
    std::memset(&output_matrix[0][0], 0, total_bytes);
    
    if(!skip_ECC)
    {
       #pragma omp parallel for
       for(int i=0; i<RS_ROWS; i++)
       {

        /*Instantiate RS Block For Codec */

        schifra::reed_solomon::block<code_length,fec_length> block;
        schifra::reed_solomon::copy(&RS_matrix[i][0], code_length, block);


        /* Transform message into Reed-Solomon encoded codeword */

        if (!encoder.encode(block))
        {
          std::cout << "Error - Critical encoding failure! "
                 << "Msg: " << block.error_as_string()  << std::endl;
        //   return 1;
        }
     
        for(int j=0; j<code_length; j++){
          uint16_t temp  = static_cast<uint16_t>(block.data[j] & 0xFFFF);
          output_matrix[i][2*j]   = temp >> 8;
          output_matrix[i][2*j+1] = temp & 0xFF;
        }
       }
    }

    else{ //skip RS
       for(int i=0; i<RS_ROWS; i++)
	{
         schifra::reed_solomon::block<code_length,fec_length> block;
         schifra::reed_solomon::copy(&RS_matrix[i][0], code_length, block);
         for(int j=0; j<data_length; j++){
            uint16_t temp  = static_cast<uint16_t>(block.data[j] & 0xFFFF);
            output_matrix[i][2*j]   = temp >> 8;
            output_matrix[i][2*j+1] = temp & 0xFF;
         }
	}
    }

    std::ofstream strand_file;
    strand_file.open(argv[2]);
    if(!strand_file){
        std::cout << "Error: output file could not be created." << std::endl;
        return 1;
    }

    int write_length = code_length;
    if(skip_ECC) write_length = data_length;
 
    for(int i=0; i<write_length; i++)
    {
      std::string s=ordToDNA(static_cast<uint16_t>(i));
          for(int j=0; j<RS_ROWS; j++) 
         s+=symbolToDNA(static_cast<unsigned char>(output_matrix[j][2*i]), static_cast<unsigned char>(output_matrix[j][2*i+1]));    
      strand_file<<s<<std::endl;    
    }
    strand_file.close();
    return 0;
}

