#include "Ordered_list.h"

Error_code Ordered_list::insert(const Record &data)
{
	int s = size();
	int position;
	for(position =0; position<s; position++){
		Record list_data;
		retrieve(position, list_data);
		if(data>=list_data)break;
	}
	return CList<Record>::insert(position,data);
}
Error_code Ordered_list::insert(int position, const Record &data)
{
	Record list_data;
	if(position>0){
		retrieve(position-1,list_data);
		if(data<list_data)
			return fail;
	}
	if(position<size()){
		retrieve(position,list_data);
		if(data>list_data)
			return fail;
	}
	return CList<Record>::insert(position,data);
}
Error_code Ordered_list::replace(int position, const Record &data)
{
	Record front, back;
	if(position==0){
		retrieve(position+1,back);
		if(data>back)
			return fail;
	}
	else if(position==size()-1){
		retrieve(position-1,front);
		if(data<front)
			return fail;
	}
	else{
		retrieve(position-1,front);
		retrieve(position+1,back);
		if(data<front || data>back)
			return fail;
	}
	return CList<Record>::replace(position,data);
}