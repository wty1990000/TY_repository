#include "Ordered_list.h"

using namespace std;

//Recursive implementation without equality checking
Error_code recursive_binary_1(const Ordered_list &the_list, const Key &target,
							  int bottom, int top, int &position)
{
	Record data;
	if(bottom<top){
		int mid = (bottom+top)/2;
		the_list.retrieve(mid, data);
		if(data<target)
			return recursive_binary_1(the_list,target,mid+1,top,position);
		else
			return recursive_binary_1(the_list,target,bottom,mid,position);
	}
	else if(top<bottom)
		return not_present;
	else{
		position = bottom;
		the_list.retrieve(bottom,data);
		if(data==target) return success;
		else return not_present;
	}
}
//Iterative version without equality checking
Error_code binary_1(const Ordered_list &the_list,const Key &target,
					int &position)
{
	Record data;
	int bottom =0, top = the_list.size()-1;
	while(bottom<top){
		int mid = (bottom+top)/2;
		the_list.retrieve(mid,data);
		if(data<target)
			bottom = mid+1;
		else
			top = mid;
	}
	if(top<bottom) return not_present;
	else{
		position = bottom;
		the_list.retrieve(bottom,data);
		if(data==target) return success;
		else return not_present;
	}
}
//Recursive version with equality checking
Error_code recursive_binary_2(const Ordered_list &the_list, const Key &target,
							  int bottom, int top, int &position)
{
	Record data;
	if(bottom<=top){
		int mid=(top+bottom)/2;
		the_list.retrieve(position,data);
		if(data==target){
			position = mid;
			return success;
		}
		else if(data<target)
			recursive_binary_2(the_list,target,mid+1,top,position);
		else
			recursive_binary_2(the_list,target,bottom,mid,position);
	}
	else return not_present;
}
Error_code binary_2(const Ordered_list &the_list, const Key &target,
					int &position)
{
	Record data;
	int bottom = 0, top = the_list.size()-1;
	while(bottom<=top){
		position = (bottom+top)/2;
		the_list.retrieve(position,data);
		if(data==target) return success;
		if(data<target) bottom=position+1;
		else top = position-1;
	}
	return not_present;
}