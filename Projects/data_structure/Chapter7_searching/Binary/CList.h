#ifndef CLIST_H_
#define CLIST_H_

#include "utilities.h"

const int max_list = 10;

template<class List_entry>
class CList{
public:
	CList(){count =0;}
	int size()const;
	bool full()const;
	bool empty()const;
	void clear();
	void traverse(void(*visit)(List_entry&));
	Error_code retrieve(int position, List_entry &x)const;
	Error_code replace(int position, const List_entry &x);
	Error_code remove(int position, List_entry &x);
	Error_code insert(int position, const List_entry &x);
protected:
	int count;
	List_entry entry[max_list];
};

template<class List_entry>
int CList<List_entry>::size()const
{
	return count;
}
template<class List_entry>
bool CList<List_entry>::full()const
{
	return count == max_list;
}
template<class List_entry>
bool CList<List_entry>::empty()const
{
	return count ==0;
}
template<class List_entry>
void CList<List_entry>::clear()
{
	count = 0;
}
template<class List_entry>
Error_code CList<List_entry>::insert(int position, const List_entry &x)
{
	if(full())
		return overflow;
	if(position<0 || position>count)
		return ranges_error;
	for(int i=count-1; i>=position; i--)
		entry[i+1] = entry[i];
	entry[position]=x;
	count++;
	return success;
}
template<class List_entry>
Error_code CList<List_entry>::remove(int position, List_entry &x)
{
	if(empty())
		return underflow;
	if(position<0 || position>=count)
		return ranges_error;
	for(int i=position;i<count;i++)
		entry[i] = entry[i+1];
	entry[position] = x;
	count--;
	return success;
}
template<class List_entry>
Error_code CList<List_entry>::retrieve(int position, List_entry &x)const
{
	if(position<0 || position>count)
		return ranges_error;
	x = entry[position];
	return success;
}
template<class List_entry>
Error_code CList<List_entry>::replace(int position, const List_entry &x)
{
	if(position<0 || position>count)
		return ranges_error;
	entry[position] = x;
	return success;
}
template<class List_entry>
void CList<List_entry>::traverse(void(*visit)(List_entry&))
{
	for(int i=0; i<count; i++)
		(*visit)(entry[i]);
}



#endif // !CLIST_H_
