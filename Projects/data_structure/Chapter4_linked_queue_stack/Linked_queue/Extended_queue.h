#ifndef EXTENDED_QUEUE_H_
#define EXTENDED_QUEUE_H_

#include "LQueue.h"

class Extended_queue : public LQueue{
public:
	bool full()const;
	int size()const;
	void clear();
	Error_code serve_and_retieve(Queue_entry &item);
};

#endif // !EXTENDED_QUEUE_H_


