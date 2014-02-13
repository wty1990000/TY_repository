#ifndef SORTABLE_LIST_H_
#define SORTABLE_LIST_H_

#include "LList.h"

template<class Record>
class LSortable_list:public LList<Record>{
public:
	void insertion_sort();
	void merge_sort();
private:
	void recursive_merge_sort(Node<Record> *&sub_list);
	Node<Record> *devide_from(Node<Record> *sub_list);
	Node<Record> *merge(Node<Record> *first, Node<Record> *second);
};
template <class Record>
void LSortable_list<Record>::insertion_sort()
{
	Node<Record>*first_unsorted,
				*last_sorted,
				*current,
				*trailing;
	if(head != nullptr){
		last_sorted = head;
		while(last_sorted->next!=nullptr){
			first_unsorted = last_sorted->next;
			if(first_unsorted->entry < head->entry){
				last_sorted->next = first_unsorted->next;
				first_unsorted->next = head;
				head = first_unsorted;
			}
			else{
				trailing = head;
				current = trailing->next;
				while(first_unsorted->entry > current->entry){
					trailing = current;
					current=trailing->next;
				}
				if(first_unsorted == current)
					last_sorted = first_unsorted;
				else{
					last_sorted->next = first_unsorted->next;
					first_unsorted->next = current;
					trailing->next = first_unsorted;
				}
			}
		}
	}
}
template <class Record>
void LSortable_list<Record>::merge_sort()
{
	recursive_merge_sort(head);
}
template<class Record>
void LSortable_list<Record>::recursive_merge_sort(Node<Record> *&sub_list)
{
	if(sub_list != nullptr && sub_list->next != nullptr){ //Not empty or contain just one entry
		Node<Record> *second = devide_from(sub_list);
		recursive_merge_sort(sub_list);
		recursive_merge_sort(second);
		sub_list = merge(sub_list,second);
	} 
}
template<class Record>
Node<Record> *LSortable_list<Record>:: devide_from(Node<Record> *sub_list)
{
	Node<Record> *position, *midpoint, *second;
	if((midpoint = sub_list)==nullptr) return nullptr;
	position = midpoint->next;
	while(position!=nullptr){
		position = position->next;
		if(position!=nullptr){
			position = position->next;
			midpoint = midpoint->next;
		}
	}
	second = midpoint->next;
	midpoint->next = nullptr;
	return second;
}
template<class Record>
Node<Record> *LSortable_list<Record>::merge(Node<Record> *first, Node<Record> *second)
{
	Node<Record> *last_sorted;
	Node<Record> combined;
	last_sorted = &combined;
	while(first!=nullptr && second!=nullptr){
		if(first->entry<=second->entry){
			last_sorted->next=first;
			last_sorted=first;
			first = first->next;
		}
		else{
			last_sorted->next = second;
			last_sorted = second;
			second = second->next;
		}
	}
	if(first == nullptr)
		last_sorted->next = second;
	else
		last_sorted->next = first;
	return combined.next;
}
#endif // !SORTABLE_LIST_H_

