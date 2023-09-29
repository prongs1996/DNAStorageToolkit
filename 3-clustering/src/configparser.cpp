#include "configparser.h"
#include <fstream>

map<string, string> config; 
string trim(string const&source, char const* delims = " \t\r\n"){
  string result(source);
  string::size_type index = result.find_last_not_of(delims);
  if(index != string::npos)
    result.erase(++index);
  index = result.find_first_not_of(delims);
  if(index != string::npos)
    result.erase(0,index);
  else
    result.erase();
  return result;
}

void ConfigParser(string const& configFile){
  ifstream file(configFile.c_str());
  string line, name, value, section;
  int posEqual, lineNo = 0;
  while(getline(file,line)){
    ++lineNo;
    if(!line.length()) continue;
    if(line[0] == '#') continue;
    if(line[0] == '['){
      if(line.find(']') != string::npos){
        section = trim(line.substr(1 , line.find(']')-1)); 
        continue;
      }
      else{
        cout << "Error while parsing config file. Line no: " << lineNo << endl;
        exit(-1);
      }
    }
    if(line.find("=") != string::npos){
      posEqual = line.find("=");
    }
    else{
      cout << "Error while parsing config file. Line no: " << lineNo << endl;  
    }
    name = trim(line.substr(0,posEqual));
    value = trim(line.substr(posEqual+1));
    config[section+"/"+name] = value;    
  }
}
string getValue(string parameter){
  if(config.find(parameter) != config.end()){
    return config[parameter];
  }
  else{
    cout << parameter << " parameter not specified in the config file" << endl;
    exit(-1);
  }
}
