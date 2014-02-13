#include "Extended_queue.h"

using namespace std;

bool Extended_queue::full() const
/*Post: if the queue is full, count = maxqueue*/
{
	return count == maxqueue;
}
int Extended_queue::size()const
/*Post: return the size of the queue*/
{
	return count;
}
void Extended_queue::clear()
{
	count = 0;
	front = 0;
	rear = maxqueue -1;
}
Error_code Extended_queue::serve_and_retrieve(Queue_entry &item)
{
	if(count<=0) return underflow;
	item = entry[front];
	count--;
	front = ((front+1 == maxqueue)?0:front+1);

	return success;
}
void Extended_queue::print_queue()
{
	if(count ==0)
		cout<<"Queue is empty"<<endl;
	else
		for(int i = rear-front; i>=0; i--)

}
