#ifndef LARRAY_H_
#define LARRAY_H_

#include "utilities.h"

typedef int index;
const int max_list = 7;

template<class List_entry>
struct Node{
public:
	List_entry entry;
	index next;
};

template<class List_entry>
class LArray{
public:
	LArray();
	int size() const;
	bool full() const;
	bool empty()const;
	void clear();
	void traverse(void(*visit)(List_entry &));

	Error_code retrieve(int position, List_entry &x)const;
	Error_code replace(int position, const List_entry &x);
	Error_code remove(int position, List_entry &x);
	Error_code insert(int position, const List_entry &x);
protected:
	Node<List_entry> workspace[max_list];
	index available,last_used,head;
	int count;

	//Auxiliary functions
	index new_node();
	void delete_node(index n);
	int current_position(index n)const;
	index set_position(int position)const;
};

template<class List_entry>
index LArray<List_entry>::new_node()
{
	index new_index;
	if(available != -1){
		new_index = available;
		available = workspace[available].next;
	}
	else if(last_used<max_list-1){
		new_index = ++last_used;
	}else return -1;
	workspace[new_index].next = -1;
	return new_index;
}
template<class List_entry>
void LArray<List_entry>::delete_node(index n)
{
	index previous;
	if(n == head) head = workspace[n].next;
	else{

	}
}
#endif // !LARRAY_H_
