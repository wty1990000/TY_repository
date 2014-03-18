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
int getyear(long int& timestamp)
{
    time_t rawtime=(time_t)timestamp;
    struct tm  ts;
    ts=*localtime(&rawtime);
	return ts.tm_year;
}
int getmonth(long int& timestamp)
{
    time_t rawtime=(time_t)timestamp;
    struct tm  ts;
    ts=*localtime(&rawtime);
	return ts.tm_mon;
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
void ouput(vector<int>& dorh, vector<int>& mon)
{
	ofstream file3("T_Y_M.csv");
	if(file3.is_open()){
		for(unsigned int i=0; i<dorh.size(); i++){
			file3<<dorh[i]<<","<<mon[i]<<endl;
		}
	}
	file3.close();
}

int main()
{
	vector<long int> timeStamp;
	vector<int>year;
	vector<int>mon;

	single_loader(timeStamp,"timeS.txt");

	for(size_t i=0; i<timeStamp.size(); i++){
		year.push_back(getyear(timeStamp[i]));
		mon.push_back(getmonth(timeStamp[i]));
	}
	ouput(year,mon);


	return 0;
}