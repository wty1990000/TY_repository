#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

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
	vector<vector<int>> wait_for_sort;
	vector<int> use_for_sort;
	vector<vector<int>> rank;
	vector<int> temp_rank;
	
	int year = 1961;

	//ofstream file("Rank.txt");
	ofstream file("Rice.txt");

	per_year_loader(wait_for_sort,"Data.csv");
	
	for(unsigned int i=0; i<wait_for_sort.size(); i++){
		use_for_sort = wait_for_sort[i]; 
		sort(use_for_sort.begin(), use_for_sort.end());
		for(unsigned int j=0; j<use_for_sort.size(); j++){
			temp_rank.push_back(find(use_for_sort.begin(),use_for_sort.end(),wait_for_sort[i][j])-use_for_sort.begin());
		}
		rank.push_back(temp_rank);
		temp_rank.resize(0);
	}

	/*if(file.is_open()){
		for(unsigned int i = 0; i< rank.size(); i++){
			file<<"\""<<year<<"\""<<": "<<"{"<<endl;
				file<<"\t"<<"\""<<"China"<<"\""<<": "<<(12-rank[i][0])<<","<<endl;
				file<<"\t"<<"\""<<"India"<<"\""<<": "<<(12-rank[i][1])<<","<<endl;
				file<<"\t"<<"\""<<"Indonesia"<<"\""<<": "<<(12-rank[i][2])<<","<<endl;
				file<<"\t"<<"\""<<"Japan"<<"\""<<": "<<(12-rank[i][3])<<","<<endl;
				file<<"\t"<<"\""<<"Korea"<<"\""<<": "<<(12-rank[i][4])<<","<<endl;
				file<<"\t"<<"\""<<"Lao"<<"\""<<": "<<(12-rank[i][5])<<","<<endl;
				file<<"\t"<<"\""<<"Malaysia"<<"\""<<": "<<(12-rank[i][6])<<","<<endl;
				file<<"\t"<<"\""<<"Myanmar"<<"\""<<": "<<(12-rank[i][7])<<","<<endl;
				file<<"\t"<<"\""<<"Philippines"<<"\""<<": "<<(12-rank[i][8])<<","<<endl;
				file<<"\t"<<"\""<<"SriLanka"<<"\""<<": "<<(12-rank[i][9])<<","<<endl;
				file<<"\t"<<"\""<<"Thailand"<<"\""<<": "<<(12-rank[i][10])<<","<<endl;
				file<<"\t"<<"\""<<"VietNam"<<"\""<<": "<<(12-rank[i][11])<<endl;
			file<<"},"<<endl<<endl;
			year++;
		}	
	}*/
	if(file.is_open()){
		for(unsigned int i = 0; i< wait_for_sort.size(); i++){
			file<<"\""<<year<<"\""<<": "<<"{"<<endl;
				file<<"\t"<<"\""<<"China"<<"\""<<": "<<wait_for_sort[i][0]<<","<<endl;
				file<<"\t"<<"\""<<"India"<<"\""<<": "<<wait_for_sort[i][1]<<","<<endl;
				file<<"\t"<<"\""<<"Indonesia"<<"\""<<": "<<wait_for_sort[i][2]<<","<<endl;
				file<<"\t"<<"\""<<"Japan"<<"\""<<": "<<wait_for_sort[i][3]<<","<<endl;
				file<<"\t"<<"\""<<"Korea"<<"\""<<": "<<wait_for_sort[i][4]<<","<<endl;
				file<<"\t"<<"\""<<"Lao"<<"\""<<": "<<wait_for_sort[i][5]<<","<<endl;
				file<<"\t"<<"\""<<"Malaysia"<<"\""<<": "<<wait_for_sort[i][6]<<","<<endl;
				file<<"\t"<<"\""<<"Myanmar"<<"\""<<": "<<wait_for_sort[i][7]<<","<<endl;
				file<<"\t"<<"\""<<"Philippines"<<"\""<<": "<<wait_for_sort[i][8]<<","<<endl;
				file<<"\t"<<"\""<<"SriLanka"<<"\""<<": "<<wait_for_sort[i][9]<<","<<endl;
				file<<"\t"<<"\""<<"Thailand"<<"\""<<": "<<wait_for_sort[i][10]<<","<<endl;
				file<<"\t"<<"\""<<"VietNam"<<"\""<<": "<<wait_for_sort[i][11]<<endl;
			file<<"},"<<endl<<endl;
			year++;
		}	
	}
				
	
	file.close();

	return 0;
}