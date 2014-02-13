#include "LList.h"

using namespace std;

template<class List_entry>
void visit(List_entry &x)
{
	cout<<x;
}

int main()
{
	LList<int> flist;
	int x;
	flist.insert(0,1);
	flist.insert(0,2);
	flist.insert(1,3);
	LList<int> slist(flist);
	flist.traverse(visit);
	slist.traverse(visit);
	cout<<endl<<flist.size();
	flist.remove(1,x);
	cout<<endl<<"remove "<<x<<" from position 1"<<endl;
	flist.traverse(visit);
	flist.clear();
	flist.insert(0,1);
	flist.traverse(visit);
}