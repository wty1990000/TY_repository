#ifndef TQUEUE_H_
#define TQUEUE_H_

#include "utilities.h"

const int maxqueue = 10;
typedef char Queue_entry;

class TQueue{
public:
	TQueue();
	bool empty() const;
	Error_code append(const Queue_entry &item);
	Error_code serve();
	Error_code retrieve(Queue_entry &item) const;
protected:
	int count;
	int front, rear;
	Queue_entry entry[maxqueue];
};


#endif // !TQUEUE_H_
