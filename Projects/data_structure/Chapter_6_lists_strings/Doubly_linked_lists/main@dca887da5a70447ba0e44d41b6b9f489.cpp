#include "DLList.h"

using namespace std;

template<class List_entry>
void visit(List_entry &x)
{
	cout<<x<<' ';
}

int main()
{
	DLList<int> flist,slist;
	int x;
	flist.insert(0,1);
	flist.insert(0,2);
	flist.insert(1,3);
	slist = flist;
	flist.traverse(visit);
	cout<<endl;
	slist.traverse(visit);
	DLList<int> tlist(flist);
	tlist.traverse(visit);
	cout<<endl<<flist.size();
	flist.remove(1,x);
	cout<<endl<<"remove "<<x<<" from position 1"<<endl;
	flist.traverse(visit);
	flist.clear();
	flist.insert(0,1);
	cout<<endl;
	flist.traverse(visit);
}