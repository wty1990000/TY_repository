#include <math.h>
#include <time.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <algorithm>

using namespace std;

int gethour(long int& timestamp)
{
    time_t rawtime=(time_t)timestamp;
    struct tm  ts;
    ts=*localtime(&rawtime);
    return ts.tm_hour; 
}
int getday(long int& timestamp)
{
    time_t rawtime=(time_t)timestamp;
    struct tm  ts;
    ts=*localtime(&rawtime);
	return ts.tm_wday; 
}
void single_loader(vector<long int>& userID,const char* filename)
{
	fstream infile(filename);
	while(infile)
	{
		string s;
		if(!getline(infile,s,','))break;

		userID.push_back(strtol(s.c_str(),NULL,10));
	}
	if(!infile.eof())
	{
		cerr<<"Fooey!\n";
	}
}
void ouput(vector<int>& dorh)
{
	ofstream file3("T_D.csv");
	if(file3.is_open()){
		for(unsigned int i=0; i<dorh.size(); i++){
			file3<<dorh[i]<<","<<endl;
		}
	}
	file3.close();
}

int main()
{
	vector<long int> timeStamp;
	vector<int>day;

	single_loader(timeStamp,"timeS.txt");

	for(size_t i=0; i<timeStamp.size(); i++){
		day.push_back(getday(timeStamp[i]));
	}
	ouput(day);


	return 0;
}