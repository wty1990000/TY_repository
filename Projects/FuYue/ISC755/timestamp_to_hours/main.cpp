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
void ouput(vector<int>& hours)
{
	ofstream file3("T.csv");
	if(file3.is_open()){
		for(unsigned int i=0; i<hours.size(); i++){
			file3<<hours[i]<<","<<endl;
		}
	}
	file3.close();
}

int main()
{
	vector<long int> timeStamp;
	vector<int>hours;

	single_loader(timeStamp,"timeS.txt");

	for(size_t i=0; i<timeStamp.size(); i++){
		hours.push_back(gethour(timeStamp[i]));
	}
	ouput(hours);


	return 0;
}