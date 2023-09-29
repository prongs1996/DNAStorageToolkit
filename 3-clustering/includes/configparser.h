#ifndef __CONFIGPARSER_H__
#define __CONFIGPARSER_H__

#include<string>
#include<map>
#include<iostream>
using namespace std;

extern map<string, string> config; 
string trim(string const& source, char const* delims);
void ConfigParser(string const& configFile);
string getValue(string parameter);

#endif


