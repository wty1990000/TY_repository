#include "LQueue.h"

using namespace std;

LQueue::LQueue()
	:front(nullptr),rear(nullptr)
/*Initialize the queue to empty.*/
{}

bool LQueue::empty()const
{
	return (front == nullptr);
}
Error_code LQueue::append(const Queue_entry &item)
/*Post: Add one new node at the rear of the queue.*/
{
	Node *new_rear = new Node(item);
	if(new_rear == nullptr) return overflow;
	if(rear == nullptr) rear = front = new_rear;
	else{
		rear->next = new_rear;
		rear = new_rear;
	}
	return success;
}
Error_code LQueue::serve()
/*Post: Serve the head node.*/
{
	if(front == nullptr) return underflow;
	Node *old_front = front;
	front = old_front->next;
	if(front == nullptr) rear = nullptr;
	delete old_front;
	return success;
}
Error_code LQueue::retrieve(Queue_entry &item)const
/*Post: Get the head of the queue.*/
{
	if(front == nullptr) return underflow;
	item = front->entry;
	return success;
}
void LQueue::print_queue()const
{
	Node *temp = front;
	if(!empty()){
		cout<<endl<<"The entries of the queue is: "<<flush;
		while(temp != nullptr){
			cout<<temp->entry<<' ';
			temp = temp->next;
		}
	}
	else
		cout<<endl<<"The queue is empty."<<flush;
}

LQueue::LQueue(const LQueue &original)
/*Post: Copy constructor*/
{
	Node *new_front, *new_rear, *original_rear= original.rear, 
		*original_front = original.front;
	if(original_front == nullptr) front = rear = nullptr;
	else{
		front = new_front = new Node(original_front->entry);
		rear  = new_rear  = new Node(original_rear->entry);
		while(original_front->next != nullptr){
			original_front = original_front->next;
			new_front->next = new Node(original_front->entry);
			new_front = new_front->next;
		}
	}
}
LQueue::~LQueue()
/*Post: Delete each queue object after scope.*/
{
	while(!empty())
		serve();
}
LQueue& LQueue::operator=(const LQueue &original)
/*Post: Overloaded = operator.*/
{
	Node *new_front, *new_copy, *new_rear, *original_rear=original.rear,
		*original_front = original.front;
	if(original_front == nullptr) new_front = new_rear = nullptr;
	else{
		new_copy = new_front = new Node(original_front->entry);
		new_rear = new Node(original_rear->entry);
		while(original_front->next != nullptr){
			original_front = original_front->next;
			new_copy->next = new Node(original_front->entry);
			new_copy = new_copy->next;
		}
	}
	while(!empty())
		serve();
	front = new_front;
	rear = new_rear;

	return *this;
}
