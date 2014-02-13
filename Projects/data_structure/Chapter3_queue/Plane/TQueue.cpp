#include "TQueue.h"

TQueue::TQueue()
/*Post: The queue is initialized*/
{
	count = 0;
	rear = maxqueue -1;
	front = 0;
}
bool TQueue::empty() const
/*Post: If the queue is empty, return true*/
{
	return count == 0;
}
Error_code TQueue::append(const Queue_entry &item)
/*Post: if the queue is not full, append a new item*/
{
	if(count>=maxqueue)return overflow;
	count++;
	rear = ((rear+1) == maxqueue)? 0: (rear+1);
	entry[rear] = item;
	return success;
}
Error_code TQueue::serve()
/*Post: if the queue is not empty, delete one item from front*/
{
	if(count<=0) return underflow;
	count--;
	front = ((front+1) == maxqueue)? 0: (front+1);
	return success;
}
Error_code TQueue::retrieve(Queue_entry &item)const
/*Post: if the queue is not empty, get the front item*/
{
	if(count<=0) return underflow;
	item = entry[front];
	return success;
}