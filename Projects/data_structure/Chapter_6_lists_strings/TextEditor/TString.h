#ifndef TSTRING_H_
#define TSTRING_H_

#include "utilities.h"
#include "DLList.h"

class TString{
public:
	TString();
	~TString();
	TString(const TString &copy); //copy constructor
	TString(const char *copy);    //covert from C-string
	TString(DLList<char>&copy);   //covert from DLList
	TString &operator=(const TString &copy);
	const char *c_str()const;	  //converstion to c-string
protected:
	char *entries;
	int length;
};

bool operator==(const TString &first, const TString &second);
bool operator> (const TString &first, const TString &second);
bool operator< (const TString &first, const TString &second);
bool operator>=(const TString &first, const TString &second);
bool operator<=(const TString &first, const TString &second);
bool operator!=(const TString &first, const TString &second);

TString read_in(std::istream &input, int &terminator);
void write(TString &s);

#endif // !TSRING_H_

