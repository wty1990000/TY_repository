#include "CList.h"

using namespace std;

template<class List_entry>
void visit(List_entry &x)
{
	cout<<x;
}

int main()
{
	CList<int> flist;
	flist.insert(0,1);
	flist.insert(0,2);
	flist.insert(1,3);
	flist.traverse(visit);
}