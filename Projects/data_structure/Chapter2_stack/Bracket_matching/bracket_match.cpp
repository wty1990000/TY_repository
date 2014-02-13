#include "TStack.h"

using namespace std;

int main()
{
	TStack openings;
	char symbol;
	bool is_matched = true;
	while(is_matched && (symbol = cin.get()) !='\n'){
		if(symbol =='{' || symbol =='(' || symbol == '[')
			openings.push(symbol);
		if(symbol =='}' || symbol ==')' || symbol == ']'){
			if(openings.empty()){
				cout <<"Unmatched closing bracket"<<symbol
					<<" detected."<<endl;
				is_matched = false;
			}
			else{
				char match;
				openings.top(match);
				openings.pop();
				is_matched = (symbol =='}' && match == '{')
							|| (symbol ==')' && match =='(')
							|| (symbol ==']' && match =='[');
				if(!is_matched)
					cout<<"Bad match" <<match<<symbol<<endl;
			}
		}
	}
	if(!openings.empty())
		cout<<"Unmatched opening bracket(s) detected."<<endl;
}