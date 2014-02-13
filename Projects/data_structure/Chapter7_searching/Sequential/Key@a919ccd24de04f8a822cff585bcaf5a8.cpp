#include "Key.h"

int Key::comparisons=0;
typedef Key Record;

Key::Key(int x)
	:key(x)
{}
int Key::the_key()const
{
	return key;
}

bool operator==(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() == y.the_key();
}
bool operator<(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() < y.the_key();
}
bool operator>(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() > y.the_key();
}
bool operator>=(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() >= y.the_key();
}
bool operator<=(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() <= y.the_key();
}
bool operator!=(const Key &x, const Key &y)
{
	Key::comparisons++;
	return x.the_key() != y.the_key();
}