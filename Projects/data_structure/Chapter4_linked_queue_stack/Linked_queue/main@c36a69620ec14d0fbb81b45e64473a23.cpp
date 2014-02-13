#include "Extended_queue.h"

using namespace std;

int main()
{
	Queue_entry item;
	Extended_queue q;
	q.append('a');
	q.append('b');
	q.append('c');
	q.print_queue();
	cout<<q.size()<<endl;
	q.serve_and_retieve(item);
}