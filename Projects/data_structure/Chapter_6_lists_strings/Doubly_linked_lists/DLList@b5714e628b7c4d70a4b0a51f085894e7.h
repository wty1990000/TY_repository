#ifndef DLLIST_H_
#define DLLIST_H_

#include "utilities.h"
#include "Node.h"

template<class List_entry>
class DLList{
public:
	DLList();
	int size()const;
	bool full()const;
	bool empty()const;
	void clear();
	void traverse(void(*visit)(List_entry&));
	Error_code retrieve(int position, List_entry &x)const;
	Error_code replace(int position, const List_entry &x);
	Error_code remove(int position, List_entry &x);
	Error_code insert(int position, const List_entry &x);
	//The three important methods
	~DLList();
	DLList(const DLList<List_entry>&copy);
	DLList<List_entry> &operator=(const DLList<List_entry>&copy);
protected:
	int count;
	mutable int current_position;
	mutable Node<List_entry> *current;
	//The following auxiliary function is used to lacate list positions
	void set_position(int position)const;
};
template<class List_entry>
void DLList<List_entry>::set_position(int position)const
{
	if(current_position<=position)
		for(; current_position != position; current_position++)
			current = current->next;
	else
		for(; current_position !=position; current_position--)
			current = current->back;
}
template<class List_entry>
DLList<List_entry>::DLList()
	:count(0),current_position(0),current(nullptr)
{}
template<class List_entry>
int DLList<List_entry>::size()const
{
	return count;
}
template<class List_entry>
bool DLList<List_entry>::full()const
{
	Node<List_entry> *test = new Node<List_entry>();
	if(test == nullptr) return true;
	else return false;
}
template<class List_entry>
bool DLList<List_entry>::empty()const
{
	return count ==0;
}
template<class List_entry>
void DLList<List_entry>::clear()
{
	Node<List_entry> *forward, *backward, *temp;
	forward = current;
	backward = current->back;
	while(forward != nullptr){
		temp = forward;
		forward = forward->next;
		delete temp;
	}
	while(backward != nullptr){
		temp = backward;
		backward = backward->back;
		delete temp;
	}
	current = nullptr;
	current_position = 0;
	count =0;
}
template<class List_entry>
void DLList<List_entry>::traverse(void(*visit)(List_entry&))
{
	Node<List_entry> *head = current,*cu = current;
	while(head->back !=nullptr){
		head = head->back;
	}
	while(head !=cu){
		(*visit)(head->entry);
		head = head->next;
	}
	while(cu != nullptr){
		(*visit)(cu->entry);
		cu = cu->next;
	}
}
template<class List_entry>
Error_code DLList<List_entry>::insert(int position,const List_entry &x)
{
	Node<List_entry> *preceding, *following, *new_node;
	if(position<0 || position>count) return ranges_error;
	if(position ==0){
		if(count ==0) following = nullptr;
		else{
			set_position(0);
			following = current;
		}
		preceding = nullptr;
	}
	else{
		set_position(position-1);
		preceding = current;
		following = preceding->next;
	}
	new_node = new Node<List_entry>(x,following,preceding);
	if(new_node == nullptr) return overflow;
	if(preceding != nullptr) preceding->next = new_node;
	if(following != nullptr) following->back = new_node;
	current = new_node;
	current_position = position;
	count++;
	return success;
}
template<class List_entry>
Error_code DLList<List_entry>::remove(int position, List_entry &x)
{
	if(position<0 || position>=count) return ranges_error;
	Node<List_entry> *original;
	if(position==0){
		if(count ==0) return underflow;
		else{
			set_position(0);
			original = current;
			if(original->next != nullptr){
				current = original->next;
				current_position = 0;
				original->next->back = nullptr;
			}
			else{
				current_position =0;
				current = nullptr;
			}
		}
	}
	else if(position == count-1){
		set_position(count-1);
		original = current;
		current = original->back;
		current_position = position-1;
		original->back->next = nullptr;
	}
	else{
		set_position(position);
		original = current;
		original->back->next = original->next;
		original->next->back = original->back;
		current = original->next;
		current_position = position+1;
	}
	x = original->entry;
	delete original;
	count--;
	return success;
}
template<class List_entry>
Error_code DLList<List_entry>::retrieve(int position, List_entry &x)const
{
	if(position<0 || position>=count) return ranges_error;
	if(empty()) return underflow;
	set_position(position);
	Node<List_entry> *temp = current;
	x= current->entry;
	return success;
}
template<class List_entry>
Error_code DLList<List_entry>::replace(int position, const List_entry &x)
{
	if(position<0 || position>= count) return ranges_error;
	if(empty()) return underflow;
	set_position(position);
	Node<List_entry> *temp = current;
	temp->entry = x;
	return success;
}
template<class List_entry>
DLList<List_entry>::~DLList()
{
	if(!empty())
		clear();
}
template<class List_entry>
DLList<List_entry>::DLList(const DLList<List_entry> &copy)
{
	Node<List_entry> *new_copyb, *new_copyf, 
		*original_currb = copy.current, *original_currf = copy.current;
	if(original_currb  == nullptr){
		count =0;
		current = nullptr;
		current_position = 0;
	}
	else{
		count = copy.count;
		current_position = copy.current_position;
		current = new_copyb = new_copyf = new Node<List_entry>(original_currb->entry);
		while(original_currb->back != nullptr){
			original_currb = original_currb->back;
			new_copyb->back = new Node<List_entry>(original_currb->entry);
			new_copyb->back->next = new_copyb;
			new_copyb = new_copyb->back;
		}
		while(original_currf->next != nullptr){
			original_currf = original_currf->next;
			new_copyf->next = new Node<List_entry>(original_currf->entry);
			new_copyf->next->back = new_copyf;
			new_copyf = new_copyf->next;
		}
		
	}
}
template<class List_entry>
DLList<List_entry>& DLList<List_entry>::operator=(const DLList<List_entry> &copy)
{
	int new_count;
	int new_position;
	Node<List_entry> *new_copyb, *new_copyf, *new_current,
		*original_currb = copy.current, *original_currf = copy.current->next;
	if(original_currb == nullptr){
		new_count =0;
		new_position = 0;
		new_current = nullptr;
	}
	else{
		new_count = copy.count;
		new_position = copy.current_position;
		new_copyb = new_copyf = new_current = new Node<List_entry>(original_currb->entry);
		while(original_currb->back != nullptr){
			original_currb = original_currb->back;
			new_copyb->back = new Node<List_entry>(original_currb->entry);
			new_copyb->back->next = new_copyb;
			new_copyb = new_copyb->back;
		}
		while(original_currf->next != nullptr){
			original_currf = original_currf->next;
			new_copyf->next = new Node<List_entry>(original_currf->entry);
			new_copyf->next->back = new_copyf;
			new_copyf = new_copyf->next;
		}

	}
	if(!empty())
		clear();
	current = new_current;
	count = new_count;
	current_position = new_position;

	return *this;
}
#endif // !DLLIST_H_
