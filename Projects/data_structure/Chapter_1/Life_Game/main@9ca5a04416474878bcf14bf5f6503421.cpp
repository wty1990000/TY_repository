#include "Life.h"
#include "utilities.h"

using namespace std;

void instructions()
/*Pre: None
  Post: Print the instructions for the users*/
{
	cout<<"Welcome to Conway's Game of Life."<<endl;
	cout<<"This game uses a grid of size"
		<<maxrow<<"by"<<maxcol<<" in which"<<endl;
	cout<<"each cell can either be occupied by an organism or not."<<endl;
	cout<<"The occupied cells change from generation to genration"<<endl;
	cout<<"according to the number of neighboring cells which are alive."
		<<endl;
}

int main(void)
/*Pre: The user supplies an initial configuration of living cells
  Post: The program prints a sequence of pictures showing the changes in the copnfiguration 
        of living cells according to the rules for the game of life
  Uses: The class Life contains methods initialize(), print(), and update().
        And the functions instructions(), user_says_yes()*/
{
	Life configutaion;
	instructions();
	configutaion.initialize();
	configutaion.print();
}