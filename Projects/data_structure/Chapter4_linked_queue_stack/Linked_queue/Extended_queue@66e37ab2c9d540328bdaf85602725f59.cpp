#include "Extended_queue.h"

int Extended_queue::size()const
{
	Node *temp = front;
	int count = 0;
	while(temp != nullptr){
		temp = temp->next;
		count++;
	}
	return count;
}
bool Extended_queue::full()const
{
	Queue_entry item;
	Node *temp = new Node(item);
	if(temp == nullptr){
		std::cout<<"No enough memory"<<std::endl;
		return false;
	}
	else{
		delete temp;
		return true;
	}
}
void Extended_queue::clear()
{
	while(!empty())
		serve();
}
Error_code Extended_queue::serve_and_retieve(Queue_entry &item)
{
	if(front == nullptr) return underflow;
	item = front->entry;
	Node *old_front = front;
	front = old_front->next;
}