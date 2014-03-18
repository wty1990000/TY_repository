#include <math.h>
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

void single_loaderi(vector<int>& iD,const char* filename)
{
	fstream infile(filename);
	while(infile)
	{
		string s;
		if(!getline(infile,s,','))break;

		iD.push_back(strtol(s.c_str(),NULL,10));
	}
	if(!infile.eof())
	{
		cerr<<"Fooey!\n";
	}
}
void single_loaderf(vector<float>& lL, const char* filename)
{
	fstream infile(filename);
	while(infile)
	{
		string s;
		if(!getline(infile,s,','))break;

		lL.push_back(atof(s.c_str()));
	}
	if(!infile.eof())
	{
		cerr<<"Fooey!\n";
	}
}
void V_Check_LAT_LONG(vector<int>& V, vector<int>& V2, vector<float>& LAT, vector<float>& LONG, vector<int>& hours, int n)
{
	vector<unsigned int> checkin(V.size(),0);
	for(size_t i=0; i<hours.size(); i++){
		if(hours[i] == n){
			for(size_t j=0; j<V.size(); j++){
				if(V[j] == V2[i]){
					checkin[j]++;
				}
			}
		}
	}
	string s = to_string(n);
	//string ss = "Results/Category_2/Category_2_"+s+".csv";
	string sss = "Results/Category_1/Hour/Category_1_MultiVenue_Hour"+s+".csv";

	//ofstream file(ss.c_str());
	ofstream file1(sss.c_str());

	for(size_t i=0; i < LAT.size(); i++){
		if(checkin[i] != 0){
			/*if( i != LAT.size()-1){
				file<<V[i]<<","<<checkin[i]<<","<<LAT[i]<<","<<LONG[i]<<endl;
			}
			else
				file<<V[i]<<","<<checkin[i]<<","<<LAT[i]<<","<<LONG[i];*/
			for(int j=0; j<checkin[i]; j++){
				file1<<V[i]<<","<<LAT[i]<<","<<LONG[i]<<endl;
			}
		}
	}
	//file.close();
	file1.close();
}

int main()
{
	vector<int>V;
	vector<int>V2;
	vector<int>hours;
	vector<float>LAT;
	vector<float>LONG;

	single_loaderf(LAT,"CAT1/LAT.csv");
	single_loaderf(LONG,"CAT1/LONG.csv");
	single_loaderi(V, "CAT1/V.csv");
	single_loaderi(hours, "U_V_T/T.csv");
	single_loaderi(V2, "U_V_T/V.csv");
	vector<unsigned int> checkin(V.size(),0);

	for(int n=0; n<24; n++){
		V_Check_LAT_LONG(V,V2,LAT,LONG,hours,n);
	}

	return 0;
}