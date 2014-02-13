#include "LQueue.h"

using namespace std;

int main()
{
	LQueue q;
	q.append('a');
	q.append('b');
	q.append('c');
	q.print_queue();
	LQueue qq(q);
	qq.print_queue();
	LQueue qqq;
	qqq = q;
	qqq.print_queue();
	q.serve();
	Queue_entry a;
	q.retrieve(a);
	cout<<endl<<"The top of q is: "<<a<<endl;
	q.print_queue();
}