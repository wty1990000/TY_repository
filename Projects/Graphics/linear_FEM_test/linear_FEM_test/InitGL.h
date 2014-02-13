#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include "utilities.h"

const int iGridSize = 10;

/* --------- OpenGL initialization functions -----------*/

void initGL( int iWindowWidth, int iWindowHeight);
void initGlut(int argc, char **argv, char *pt_cWindowTitle, 
			  int iWindowWidth, int iWindowHeight, int *pt_iWindowId);

/* --------- Draw the environment -----------*/
void drawGround();
void drawAxis();