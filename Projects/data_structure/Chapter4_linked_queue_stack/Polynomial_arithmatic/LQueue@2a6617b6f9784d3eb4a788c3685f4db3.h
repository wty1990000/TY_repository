#ifndef LQUEUE_H_
#define LQUEUE_H_

#include "Node.h"
#include "utilities.h"

class LQueue{
public:
	LQueue();
	bool empty()const;
	Error_code append(const Queue_entry &item);
	Error_code serve();
	Error_code retrieve(Queue_entry &item)const;
	void print_queue()const;
	//Three important methods
	LQueue(const LQueue &original);
	~LQueue();
	LQueue& operator=(const LQueue &original);
protected:
	Node *front, *rear;
};
#endif // !LQUEUE_H_
