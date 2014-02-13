//#include "LSortable_list.h"
#include "Sortable_list.h"

using namespace std;

template<class Record>
void printC(Record &x)
{
	cout<<x<<' ';
}


int main()
{
	Sortable_list<int> L1;
	//LSortable_list<int> L1;
	L1.insert(0,3);
	L1.insert(0,1);
	L1.insert(0,2);

	L1.traverse(printC);

	L1.quick_sort();
	L1.traverse(printC);

}