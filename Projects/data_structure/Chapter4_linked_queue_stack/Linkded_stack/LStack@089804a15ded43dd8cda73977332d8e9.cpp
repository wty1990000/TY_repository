#include "LStack.h"

using namespace std;

LStack::LStack()
	:top_node(nullptr)
/*Post: Default initialize the stack to be empty.*/
{}
LStack::LStack(const LStack &original)
/*Post: copy construct the LStack*/
{
	Node *new_copy, *original_node = original.top_node;
	if(original_node == nullptr) top_node = nullptr;
	else{
		top_node = new_copy = new Node(original_node->entry);
		while(original_node->next != nullptr){
			original_node = original_node->next;
			new_copy->next = new Node(original_node->entry);
			new_copy = new_copy->next;
		}
	}
}
LStack::~LStack()
/*Post: The stack is cleared.*/
{
	while(!empty())
		pop();
}
LStack& LStack::operator=(const LStack &original)
/*Post: The stack is reset to the copy of original.*/
{
	Node *new_copy, *new_top, *original_node = original.top_node;
	if(original_node == nullptr) new_top = nullptr;
	else{
		//1. Copy the data in original
		new_copy = new_top = new Node(original_node->entry);
		while(original_node->next != nullptr){
			original_node = original_node->next;
			new_copy->next = new Node(original_node->entry);
			new_copy = new_copy->next;
		}
	}
	//2. Clear out the original entries.
	while(!empty())
		pop();
	//3. Move newly copied data to the Stack object.
	top_node = new_top;
	return *this;
}
Error_code LStack::push(const Stack_entry &item)
/*Post: Push item onto the stack; return success or overflow*/
{
	Node *new_top = new Node(item, top_node);	//new_top built and pointed to the old top
	if(new_top == nullptr) return overflow;
	top_node = new_top;							//set top_node to new_top
	return success;
}
Error_code LStack::pop()
/*Post: Pop item from the stack; return underflow if the stack is empty.*/
{
	Node *old_top = top_node;
	if(top_node == nullptr) return underflow;
	top_node = top_node->next;
	delete old_top;
	return success;
}
bool LStack::empty() const
{
	return top_node->next == nullptr;
}
Error_code LStack::top(Stack_entry &item)const
/*Post: Use item to get the top one.*/
{
	if(top_node->next == nullptr) return underflow;
	item = top_node->entry;
	return success;
}
void LStack::print_stack()const
{
	Node *temp = top_node;
	if(!empty())
		cout<<endl<<"The entries of the stack are: "<<flush;
	while(temp->next != nullptr){
		cout<<temp->entry<<'\t';
		temp = temp->next;
	}
}
