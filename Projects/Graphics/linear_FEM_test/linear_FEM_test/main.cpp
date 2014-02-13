#include "InitGL.h"
#include "physicalSimulation.h"
#include "TetMesh.h"
#include <Windows.h>

#define EPSILON 0.001f
#define EPS2  EPSILON*EPSILON

using namespace std;  

const int iWidth = 1024, iHeight = 1024;
static size_t total_points=0;
static float timeStep =  1/60.0f;
static float currentTime = 0;
static float accumulator = timeStep;
static int selected_index = -1;


int main()
{
	
	return 0;	
}

