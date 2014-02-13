#include "CList.h"
#include "Key.h"
#include "Timer.h"
#include "Random.h"

using namespace std;

Error_code sequatial_search(const CList<Record>&the_list,
							const Key &target, int &position)
{
	int s = the_list.size();
	for(position=0; position<s; position++){
		Record data;
		the_list.retrieve(position,data);
		if(data == target) return success;
	}
	return not_present;
}
void print_out(const char *x, const double &elapsed_time, 
			   const int &comparison, const int &search)
{
	cout<<x<<" Findings using time: "<<elapsed_time
		<<" to finish "<<comparison<<" comparisons"
		<<" in "<<search<<" searches"<<endl;
}
void test_search(int searches, CList<Record>&the_list)
{
	int list_size = the_list.size();
	if(searches<=0 || list_size<0){
		cout<<" Exiting test:"<<endl
			<<" The number of seraches must be positive."<<endl
			<<" The number of list entries must exceed 0."<<endl;
		return;
	}
	int i, target, found_at;
	Key::comparisons =0;
	Random number;
	Timer clock;
	for(i=0; i<searches; i++){
		target = 2*number.random_integer(0,list_size-1)+1;
		if(sequatial_search(the_list,target,found_at) ==not_present)
			cout<<"Error: Failed to find expected target "<<target<<endl;
	}
	print_out("Successful",clock.elapsedTime(),Key::comparisons,searches);
	Key::comparisons =0;
	clock.reset();
	for(i=0; i<searches; i++){
		target = 2*number.random_integer(0,list_size);
		if(sequatial_search(the_list,target,found_at) ==success)
			cout<<"Error: Found unexpected target "<<target
			<<" at "<<found_at<<endl;
	}
	print_out("Unsuccessful",clock.elapsedTime(),Key::comparisons,searches);
}


int main()
{
	CList<Record> test;
	for(int i=1; i<=10; i++)
		test.insert(i-1,i);
	test_search(5,test);
}