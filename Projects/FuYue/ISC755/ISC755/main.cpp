#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>

using namespace std;

//Release space allocated for a vector
template <typename T>
void freeAll(T &t)
{
	T tmp;
	t.swap(tmp);
}

void userID_per_venue_loader(vector<vector<long int>> &userIDperVenue,const char* filename);
void userID_loader(vector<long int>& userID,const char* filename);
void venueID_generater(vector<long int>& venueID);
void deleteNulls(vector<long int>& venueID, vector<vector<long int>>& userIDperVenue);
void uniqueIDPerVenue(vector<long int>& userID, vector<vector<long int>> &userIDPerVenue);
void computeWeight(vector<long int>& venueID,vector<vector<long int>>& userIDPerVenue);

int main()
{
	//Vectors for loading three files.
	vector<vector<long int>> userIDperVenue;
	vector<long int> userID;
	vector<long int> venueID;
	//vector<vector<long int>> uniqueUserIDPerV;

	//Get unique userIDs checking in each venue ID.
	userID_loader(userID,"A.txt");
	userID_per_venue_loader(userIDperVenue,"user_ID_null.txt");
	venueID_generater(venueID);
	deleteNulls(venueID,userIDperVenue);
	uniqueIDPerVenue(userID,userIDperVenue);

	freeAll(userID);
	
	computeWeight(venueID, userIDperVenue);

	cout<<userIDperVenue.size()<<endl;
	cout<<userID.size()<<endl;
	cout<<venueID.size()<<endl;

}

void userID_per_venue_loader(vector<vector<long int>> &userIDperVenue,const char* filename)
{
	ifstream infile(filename);
	
	while(infile)
	{
		string s;
		if(!getline(infile,s))break;

		istringstream ss(s);
		vector<long int> record;
		
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
void userID_loader(vector<long int>& userID,const char* filename)
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
void venueID_generater(vector<long int>& venueID)
{
	for (int i = 0; i<43107; i++)
		venueID.push_back(i);
}
void deleteNulls(vector<long int>& venueID, vector<vector<long int>>& userIDperVenue)
{
	vector<vector<long int>>::iterator it;
	for(it = userIDperVenue.begin(); it != userIDperVenue.end(); ++it){
		if(it->size() ==0){
			venueID.erase(venueID.begin()+distance(userIDperVenue.begin(),it));
			userIDperVenue.erase(it);
		}
	}

}
void uniqueIDPerVenue(vector<long int>& userID, vector<vector<long int>> &userIDPerVenue)
{
	vector<vector<long int>>::iterator it_UserIdperVenue;
	vector<long int>::iterator it_UserIndex;
	/*vector<long int> record;*/
	for(it_UserIdperVenue = userIDPerVenue.begin(); it_UserIdperVenue != userIDPerVenue.end(); ++it_UserIdperVenue){
		for(it_UserIndex = it_UserIdperVenue->begin(); it_UserIndex != it_UserIdperVenue->end(); ++it_UserIndex){
			*it_UserIndex = userID[*it_UserIndex];
		}
		/*set<long int>s(record.begin(),record.end());
		record.assign(s.begin(),s.end());*/
		sort(it_UserIdperVenue->begin(),it_UserIdperVenue->end());
		it_UserIdperVenue->erase(unique(it_UserIdperVenue->begin(),it_UserIdperVenue->end()),it_UserIdperVenue->end());
	}
}
void computeWeight(vector<long int>& venueID, vector<vector<long int>>& userIDPerVenue)
{
	vector<long int> V;
	vector<long int> V1;
	vector<long int> V2;
	vector<long int>::iterator itInner;

	ofstream file("ISC_NEW_test.csv");
	if(file.is_open()){
		int l = 1;

		for(int i = venueID.size()-1; i>0; i--){
			for(int m = l; m<=l-1+i; m++)
			{
				int numer = 0; float weight=0;
				V1 = userIDPerVenue[venueID.size()-i-1];
				V2 = userIDPerVenue[venueID.size()-i+m-l];
				set_intersection(V1.begin(),V1.end(),V2.begin(),V2.end(),back_inserter(V));
				/*V.resize(itInner-V.begin());*/
				numer = V.size();
				freeAll(V);
				weight = float(numer)/float(userIDPerVenue[venueID.size()-i-1].size() + userIDPerVenue[venueID.size()-i+m-l].size()-numer);
				if(weight>=0.3)
					file<<venueID[venueID.size()-i-1]<<","<<venueID[venueID.size()-i+m-l]<<","<<weight<<","<<endl;
			}
			l++;
		}
	}
	file.close();
}