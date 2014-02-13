#include "TString.h"

using namespace std;

TString::TString()
	:entries(nullptr),length(0)
{}
TString::~TString()
{
	if(entries != nullptr)
		delete []entries;
}
TString::TString(const TString &copy)
{
	length = copy.length;
	entries = new char[length];
	strcpy_s(entries,copy.entries);
}
TString::TString(const char *copy)
{
	length = strlen(copy);
	entries = new char[length+1];
	strcpy(entries,copy);
}
TString::TString(DLList<char>&copy)
{
	length = copy.size();
	entries = new char[length+1];
	for(int i=0; i<length; i++) copy.retrieve(i,entries[i]);
	entries[length]='\0';
}
TString& TString::operator=(const TString &copy)
{
	length = copy.length;
	if(entries!=nullptr)
		delete [] entries;
	entries = new char[length];
	strcpy(entries,copy.entries);
}
const char* TString::c_str()const
{
	return (const char*)entries;
}

bool operator==(const TString &first, const TString &second)
{
	return strcmp(first.c_str(),second.c_str())==0;
}
bool operator>(const TString &first, const TString &second)
{
	return strcmp(first.c_str(),second.c_str())>0;
}
bool operator<(const TString &first, const TString &second)
{
	return strcmp(first.c_str(),second.c_str())<0;
}
bool operator>=(const TString &first, const TString &second)
{
	return (strcmp(first.c_str(),second.c_str())>0 || 
		strcmp(first.c_str(),second.c_str())==0);
}
bool operator<=(const TString &first, const TString &second)
{
	return (strcmp(first.c_str(),second.c_str())<0 || 
		strcmp(first.c_str(),second.c_str())==0);
}
bool operator!=(const TString &first, const TString &second)
{
	return strcmp(first.c_str(),second.c_str())!=0;
}

//Input and output
TString read_in(istream &input)
{
	DLList<char>temp;
	int size =0;
	char c;
	while((c=input.peek())!=EOF && (c=input.get())!='\n')
		temp.insert(size++,c);
	TString answer(temp);
	return answer;
}
void write(TString &x)
{
	cout<<x.c_str()<<endl;
}