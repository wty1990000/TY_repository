#ifndef LLIST_H_
#define LLIST_H_

#include "utilities.h"
#include "Node.h"

template<class List_entry>
class LList{
public:
	LList();
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
	~LList();
	LList(const LList<List_entry>&copy);
	LList<List_entry> &operator=(const LList<List_entry>&copy);
protected:
	int count;
	mutable int current_position;
	Node<List_entry> *head;
	mutable Node<List_entry> *current;
	//The following auxiliary function is used to lacate list positions
	Node<List_entry> *set_position(int position)const;
};
template<class List_entry>
Node<List_entry> *LList<List_entry>::set_position(int position)const
/*Pre: Poistion must be valid: 0<=position<count
  Post: returns a pointer to the Node in position;*/
{
	Node<List_entry> *q = head;
	for(int i=0; i<position;i++) q = q->next;
	return q;
}
template<class List_entry>
Error_code LList<List_entry>::insert(int position, const List_entry &x)
{
	if(position<0 || position>count)
		return ranges_error;
	Node<List_entry> *new_node, *previous, *following;
	if(position>0){
		previous = set_position(position-1);
		following = previous->next;
	}
	else{
		following = head;
		previous = nullptr;
	}
	new_node = new Node<List_entry>(x,following);
	if(new_node == nullptr)
		return overflow;
	if(position == 0)
		head = new_node;
	else
		previous->next = new_node;
	count++;
	return success;
}
template<class List_entry>
Error_code LList<List_entry>::remove(int position, List_entry &x)
{
	if(position<0 || position>=count)
		return ranges_error;
	Node<List_entry> *delete_node, *previous;
	if(empty())
		return underflow;
	if(position>0){
		delete_node = set_position(position);
		previous = set_position(position-1);
		previous->next = delete_node->next;
	}
	else{
		delete_node = head;
		head = head->next;
	}
	x = delete_node->entry;
	delete delete_node;
	count--;
	return success;
}
template<class List_entry>
Error_code LList<List_entry>::retrieve(int position, List_entry &x)const
{
	if(position<0 || position>=n)
		return ranges_error;
	if(empty())
		return underflow;
	Node<List_entry> *temp;
	temp = set_position(position);
	x = temp->entry;
	return success;
}
template<class List_entry>
Error_code LList<List_entry>::replace(int position, const List_entry &x)
{
	if(position<0 || position>=n)
		return ranges_error;
	if(empty())
		return underflow;
	Node<List_entry> *temp;
	temp = set_position(position);
	temp->entry = x;
	return success;
}
template<class List_entry>
LList<List_entry>::LList()
	:head(nullptr),count(0)
{}
template<class List_entry>
int LList<List_entry>::size()const
{
	return count;
}
template<class List_entry>
bool LList<List_entry>::full()const
{
	Node<List_entry> *test = new Node<List_entry>();
	if(test == nullptr)
		return true;
	else
		return false;
}
template<class List_entry>
bool LList<List_entry>::empty()const
{
	return count ==0;
}
template<class List_entry>
void LList<List_entry>::clear()
{
	Node<List_entry> *test;
	while(head != nullptr){
		test = head;
		head = head->next;
		delete test;
	}
}
template<class List_entry>
void LList<List_entry>::traverse(void(*visit)(List_entry&))
{
	Node<List_entry> *temp = head;
	while(temp != nullptr){
		(*visit)(temp->entry);
		temp = temp->next;
	}
}
template<class List_entry>
LList<List_entry>::~LList()
{
	if(!empty())
		clear();
}
template<class List_entry>
LList<List_entry>::LList(const LList<List_entry>&copy)
{
	Node<List_entry> *new_copy, *original_head = copy.head;
	if(original_head == nullptr) {
		count = 0;
		head = nullptr;
	}
	else{
		count = copy.count;
		head = new_copy = new Node<List_entry>(original_head->entry);
		while(original_head->next != nullptr){
			original_head = original_head->next;
			new_copy->next = new Node<List_entry>(original_head->entry);
			new_copy = new_copy->next;	
		}
	}
}
template<class List_entry>
LList<List_entry> &LList<List_entry>::operator=(const LList<List_entry>&copy)
{
	int new_count;
	Node<List_entry> &new_head, *new_copy, *original_head = copy.head;
	if(original_head == null){
		new_count =0;
		new_head = nullptr;
	}
	else{
		new_copy = new_top = new Node<List_entry>(original_head->entry);
		new_count = copy.count;
		while(original_head->next != nullptr){
			original_head = original_head->next;	
			new_copy->next = new Node<List_entry>(original_head->entry);
			new_copy = new_copy->next;
		}
	}
	if(!empty())
		clear();
	head = new_head;
	count = new_count;

	return *this;
}
#endif // !LLIST_H_
