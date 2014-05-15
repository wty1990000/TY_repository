#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

void single_col_loader(vector<int> &col, const char* filename)
{
	fstream infile(filename);
	while(infile)
	{
		string s;
		if(!getline(infile,s,','))break;

		col.push_back(strtol(s.c_str(),NULL,10));
	}
	if(!infile.eof())
	{
		cerr<<"Fooey!\n";
	}
}
void single_col_writer(const vector<int>& col, const char* filename, int val,int year)
{
	ofstream file(filename);
	
	if(file.is_open()){
		for(int i=0; i<year; i++){
			for(int j=i*val; j<i*val+val; j++){
				file<<col[j]<<",";
			}
			file<<endl;
		}		
	}
	file.close();
}
void per_year_loader(vector<vector<int>> &userIDperVenue,const char* filename)
{
	ifstream infile(filename);
	
	while(infile)
	{
		string s;
		if(!getline(infile,s))break;

		istringstream ss(s);
		vector<int> record;
		
		while(ss)
		{
			string s;
			if(!getline(ss,s,','))break;
			record.push_back(strtol(s.c_str(),NULL,10));
		}
		userIDperVenue.push_back(record);
	}
	if(!infile.eof())
	{
		cerr<<"Fooey!\n";
	}
}

int main()
{
	vector<int> col;
	vector<vector<int>> wait_for_sort;
	vector<int> use_for_sort;
	vector<vector<int>> rank;
	vector<int> temp_rank;
	
	single_col_loader(col,"production/p_number_large.csv");
	single_col_writer(col,"production/p_number_l.csv",20,50);

	int year = 1961;

	ofstream file_orank("production/p_orank.csv");
	ofstream file_rank("production/p_Rank.txt");
	ofstream file("production/p_json.txt");

	per_year_loader(wait_for_sort,"production/p_number_l.csv");
	
	for(unsigned int i=0; i<wait_for_sort.size(); i++){
		use_for_sort = wait_for_sort[i]; 
		sort(use_for_sort.begin(), use_for_sort.end());
		for(unsigned int j=0; j<use_for_sort.size(); j++){
			temp_rank.push_back(find(use_for_sort.begin(),use_for_sort.end(),wait_for_sort[i][j])-use_for_sort.begin());
		}
		rank.push_back(temp_rank);
		temp_rank.resize(0);
	}

	if(file_orank.is_open()){
		for(unsigned int i = 0; i< rank.size(); i++){
			for(unsigned int j=0; j<rank[i].size(); j++){
				file_orank<<(20-rank[i][j])<<endl;
			}
		}
	}
	file_orank.close();

	if(file_rank.is_open()){
		for(unsigned int i = 0; i< rank.size(); i++){
			file_rank<<"\""<<year<<"\""<<": "<<"{"<<endl;
				file_rank<<"\t"<<"\""<<"Bangladesh"<<"\""<<": "<<(20-rank[i][0])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Brazil"<<"\""<<": "<<(20-rank[i][1])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Cambodia"<<"\""<<": "<<(20-rank[i][2])<<","<<endl;
				file_rank<<"\t"<<"\""<<"China"<<"\""<<": "<<(20-rank[i][3])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Democratic People's Republic of Korea"<<"\""<<": "<<(20-rank[i][4])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Egypt"<<"\""<<": "<<(20-rank[i][5])<<","<<endl;
				file_rank<<"\t"<<"\""<<"India"<<"\""<<": "<<(20-rank[i][6])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Indonesia"<<"\""<<": "<<(20-rank[i][7])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Japan"<<"\""<<": "<<(20-rank[i][8])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Madagascar"<<"\""<<": "<<(20-rank[i][9])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Malaysia"<<"\""<<": "<<(20-rank[i][10])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Myanmar"<<"\""<<": "<<(20-rank[i][11])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Nepal"<<"\""<<": "<<(20-rank[i][12])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Pakistan"<<"\""<<": "<<(20-rank[i][13])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Philippines"<<"\""<<": "<<(20-rank[i][14])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Republic of Korea"<<"\""<<": "<<(20-rank[i][15])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Sri Lanka"<<"\""<<": "<<(20-rank[i][16])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Thailand"<<"\""<<": "<<(20-rank[i][17])<<","<<endl;
				file_rank<<"\t"<<"\""<<"United States of America"<<"\""<<": "<<(20-rank[i][18])<<","<<endl;
				file_rank<<"\t"<<"\""<<"Viet Nam"<<"\""<<": "<<(20-rank[i][19])<<","<<endl;
			file_rank<<"},"<<endl<<endl;
			year++;
		}	
	}
	file_rank.close();

	year = 1961;

	if(file.is_open()){
		for(unsigned int i = 0; i< wait_for_sort.size(); i++){
			file<<"\""<<year<<"\""<<": "<<"{"<<endl;
				file<<"\t"<<"\""<<"Bangladesh"<<"\""<<": "<<wait_for_sort[i][0]<<","<<endl;
				file<<"\t"<<"\""<<"Brazil"<<"\""<<": "<<wait_for_sort[i][1]<<","<<endl;
				file<<"\t"<<"\""<<"Cambodia"<<"\""<<": "<<wait_for_sort[i][2]<<","<<endl;
				file<<"\t"<<"\""<<"China"<<"\""<<": "<<wait_for_sort[i][3]<<","<<endl;
				file<<"\t"<<"\""<<"Democratic People's Republic of Korea"<<"\""<<": "<<wait_for_sort[i][4]<<","<<endl;
				file<<"\t"<<"\""<<"Egypt"<<"\""<<": "<<wait_for_sort[i][5]<<","<<endl;
				file<<"\t"<<"\""<<"India"<<"\""<<": "<<wait_for_sort[i][6]<<","<<endl;
				file<<"\t"<<"\""<<"Indonesia"<<"\""<<": "<<wait_for_sort[i][7]<<","<<endl;
				file<<"\t"<<"\""<<"Japan"<<"\""<<": "<<wait_for_sort[i][8]<<","<<endl;
				file<<"\t"<<"\""<<"Madagascar"<<"\""<<": "<<wait_for_sort[i][9]<<","<<endl;
				file<<"\t"<<"\""<<"Malaysia"<<"\""<<": "<<wait_for_sort[i][10]<<","<<endl;
				file<<"\t"<<"\""<<"Myanmar"<<"\""<<": "<<wait_for_sort[i][11]<<endl;
				file<<"\t"<<"\""<<"Nepal"<<"\""<<": "<<wait_for_sort[i][12]<<","<<endl;
				file<<"\t"<<"\""<<"Pakistan"<<"\""<<": "<<wait_for_sort[i][13]<<","<<endl;
				file<<"\t"<<"\""<<"Philippines"<<"\""<<": "<<wait_for_sort[i][14]<<","<<endl;
				file<<"\t"<<"\""<<"Republic of Korea"<<"\""<<": "<<wait_for_sort[i][15]<<","<<endl;
				file<<"\t"<<"\""<<"Sri Lanka"<<"\""<<": "<<wait_for_sort[i][16]<<","<<endl;
				file<<"\t"<<"\""<<"Thailand"<<"\""<<": "<<wait_for_sort[i][17]<<","<<endl;
				file<<"\t"<<"\""<<"United States of America"<<"\""<<": "<<wait_for_sort[i][18]<<","<<endl;
				file<<"\t"<<"\""<<"Viet Nam"<<"\""<<": "<<wait_for_sort[i][19]<<","<<endl;
			file<<"},"<<endl<<endl;
			year++;
		}	
	}
				

	file.close();

	return 0;
}