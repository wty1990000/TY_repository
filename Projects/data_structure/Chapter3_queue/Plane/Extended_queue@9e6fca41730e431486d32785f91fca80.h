#ifndef EXTENDED_QUEUE_H_
#define EXTENDED_QUEUE_H_

#include "TQueue.h"

class Extended_queue : public TQueue{
public:
	bool full() const;
	int size() const;
	void clear();
	Error_code serve_and_retrieve(Queue_entry &item);
};

#endif // !EXTENDED_QUEUE_H_
