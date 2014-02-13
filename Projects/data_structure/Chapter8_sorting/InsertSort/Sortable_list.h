#ifndef SORTABLE_LIST_H_
#define SORTABLE_LIST_H_

#include "CList.h"

template<class Record>
class Sortable_list:public CList<Record>{
public:
	void insertion_sort();
	void selection_sort();
	void shell_sort();
	void quick_sort();
private:
	int max_key(int low, int high);
	void swap_entry(int low, int high);
	void sort_interval(int start, int increment); //modified insertion sort
	void recursive_quick_sort(int low, int high);
	int partition(int low, int high);
};

template<class Record>
void Sortable_list<Record>::insertion_sort()
{
	int first_unsorted;		//	position of first unsorted entry
	int position;			//	searches sorted part of the list
	Record current;			//	holds the entries temporarily removed from the list
	for(first_unsorted =1; first_unsorted<count; first_unsorted++){
		if(entry[first_unsorted]<entry[first_unsorted-1]){
			position = first_unsorted;
			current = entry[first_unsorted];
			do{
				entry[position]=entry[position-1];
				position--;
			}while(position>0 && entry[position-1]>current);
			entry[position]=current;
		}
	}
}
template<class Record>
void Sortable_list<Record>::selection_sort()
{
	for(int position = count-1; position>0; position--){
		int maxk = max_key(0,position);
		swap_entry(maxk,position);
	}
}
template<class Record>
void Sortable_list<Record>::shell_sort()
{
	int start, increment;
	increment = count;
	do{
		increment = increment/3+1;
		for(start = 0; start<increment; start++)
			sort_interval(start,increment);
	}while(increment>1);
}
template<class Record>
void Sortable_list<Record>::quick_sort()
{
	recursive_quick_sort(0, count-1);
}

template<class Record>
int Sortable_list<Record>::max_key(int low, int high)
{
	int maxone, current;
	maxone = low;
	for(current = low+1; current<=high; current++)
		if(entry[maxone]<entry[current])
			maxone = current;
	return maxone;
}
template<class Record>
void Sortable_list<Record>::swap_entry(int low, int high)
{
	Record temp;
	temp = entry[low];
	entry[low] = entry[high];
	entry[high] = temp;
}
template<class Record>
void Sortable_list<Record>::sort_interval(int start, int increment)
{
	int first_unsorted, position;
	Record current;
	for(first_unsorted=start+1; first_unsorted<count; first_unsorted += increment){
		if(entry[first_unsorted]<entry[first_unsorted-1]){
			position = first_unsorted;
			current = entry[first_unsorted];
			do{
				entry[position] = entry[position-1];
				position--;
			}while(position>start && entry[position-1]>current);
			entry[position]=current;
		}
	}
}
template<class Record>
void Sortable_list<Record>::recursive_quick_sort(int low, int high)
{
	int pivot_position;
	if(low < high){
		pivot_position = partition(low, high);
		recursive_quick_sort(low,pivot_position-1);
		recursive_quick_sort(pivot_position+1,high);
	}
}
template<class Record>
int Sortable_list<Record>::partition(int low ,int high)
{
	int last_small;
	Record pivot;
	
	swap_entry(low, (low+high)/2);
	pivot = entry[low];
	last_small = low;
	for(int i=low+1; i<=high; i++){
		if(entry[i]<pivot){
			last_small += 1;
			swap_entry(last_small,i);
		}
	}
	swap_entry(low,last_small);
	return last_small;
}

#endif // !SORTABLE_LIST_H_


