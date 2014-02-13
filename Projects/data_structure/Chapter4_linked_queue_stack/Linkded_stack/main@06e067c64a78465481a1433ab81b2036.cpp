#include "LStack.h"

using namespace std;

int main()
{
	LStack test;
	test.push('a');
	test.push('b');
	test.push('c');
	test.pop();
	test.print_stack();
	test.print_stack();
	
}