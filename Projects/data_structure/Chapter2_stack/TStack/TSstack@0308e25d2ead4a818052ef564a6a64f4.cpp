#include "TStack.h"

Error_code TStack::push(const stack_entry &item)
/*Pre: None
  Post: If the TStack is not full, item is added to the top 
        of the TStack. If TStack is full, Error_code of overflow
		will be returned and it's unchaged.*/
{
	Error_code outcome = success;
	if(count>=maxstack)
		outcome = overflow;
	else
		entry[count++] = item;
	return outcome;
}

Error_code TStack::pop()
/*Pre:  None
  Post: If the stack is not empty, the top of the stack is removed. 
        If the stack is empty, Erro_code of underflow will return*/
{
	Error_code outcome = success;
	if(count == 0)					//May be finished by empty();
		outcome = underflow;
	else
		--count;
	return outcome;
}

Error_code TStack::top(stack_entry &item) const
/*Pre:  None
  Post: If the stack is not empty, item takes the top of the stack.
        If it's empty, return Error_code of underflow.*/
{
	Error_code outcome = success;
	if(count ==0)
		outcome = underflow;
	else
		item = entry[count-1];
	return outcome;
}
bool TStack::empty() const
/*Pre:  None
  Post: If the stack is empty, true is returned; Otherwise return false.*/
{
	bool outcome = true;
}