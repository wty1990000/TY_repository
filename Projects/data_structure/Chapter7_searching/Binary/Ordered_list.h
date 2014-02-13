#ifndef ORDERED_LIST_H_
#define ORDERED_LIST_H_

#include "CList.h"
#include "Key.h"

class Ordered_list:public CList<Record>{
public:
	Error_code insert(const Record &data);
	Error_code insert(int position, const Record &data);
	Error_code replace(int position, const Record &data);
};

#endif // !ORDERED_LIST_H_
