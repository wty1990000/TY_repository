#include "InitGL.h"
#include "physicalSimulation.h"
#include "TetMesh.h"
#include "conjugate_gradient.h"
#include "sysinfo.h"
#include <Windows.h>

#define EPSILON 0.001f
#define EPS2  EPSILON*EPSILON

using namespace std;  

const int iWidth = 1024, iHeight = 1024;

static float timeStep =  1/60.0f;
static float currentTime = 0;
static float accumulator = timeStep;
static int selected_index = -1;

/* --------- timers -----------*/
LARGE_INTEGER frequency;        // ticks per second
LARGE_INTEGER t1, t2;           // ticks
static float frameTimeQP=0;
static float frameTime =0 ;
static float startTime =0, fps=0;
static int totalFrames=0;


/* --------- functions for GLUT -----------*/
void displayFunc(void);
void reshapeFunc(int iWindowWidth, int iWindowHeight);
void idleFunc(void);
void keyboardFunc(unsigned char key, int x, int y);
void onMouseDown(int button, int state, int x, int y);
void onMouseMotion(int x, int y);

/* --------- functions for physics-----------*/
void initilization();
void UpdateOrientation();
void stepPhysics();



int main()
{
	initializePhysics();
	initialize_CGsolver();
	initGL(iWidth, iHeight);
	sysinfoglobals = new Sysinfo();

	return 0;	
}

void onMouseDown(int button, int state, int x, int y)
{
	int iWindowX, iWindowY;
	if( GLUT_DOWN == state){
		graphicalglobals->iOldX = x;
		graphicalglobals->iOldY = y;
		iWindowY = (iHeight - y);
		iWindowX = x;
		
		float iWindowZ = 0.0;
		glReadPixels(x, iHeight - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &iWindowZ);
		double dOldX = 0.0, dOldY =0.0, dOldZ = 0.0;
		gluUnProject(iWindowX, iWindowY,iWindowZ, 
			graphicalglobals->dModelView,graphicalglobals->dPorjection,
			graphicalglobals->iViewport,&dOldX, &dOldY,&dOldZ);
		glm::vec3 pt(dOldX, dOldY, dOldZ);

		printf_s("\nObject [%3.3f, %3,3f, %3,3f]",dOldX,dOldY,dOldZ);
		size_t i=0;
		for(int i=0; i<physicalglobals->total_points; i++){
			if(glm::distance(physicalglobals->P[i],pt)<0.01){
				selected_index =i;

				printf_s("Point %d is picked:\n",i);
				printf_s("Pt [%3.3f, %3.3f, %3.3f]\n",physicalglobals->P[i].x,physicalglobals->P[i].y,physicalglobals->P[i].z);
				break;	
			}
		}
	}
	if(GLUT_MIDDLE_BUTTON == button)
		graphicalglobals->iState = 0;
	else
		graphicalglobals->iState = 1;
	if(GLUT_UP == state){
		selected_index = -1;
		UpdateOrientation();
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}
void onMouseMotion(int x, int y)
{
	if(selected_index == -1){
		if(graphicalglobals->iState == 0)
			graphicalglobals->fDist *= (1 + (y-graphicalglobals->iOldY)/60.0);
		else{
			graphicalglobals->rX +=(x - graphicalglobals->iOldX)/5.0f;
			graphicalglobals->rY +=(y - graphicalglobals->iOldY)/5.0f;
		}
	}
	else{
		float fDelta = 1000/abs(graphicalglobals->fDist);
		float fValX = (x - graphicalglobals->iOldX)/fDelta;
		float fValY = (graphicalglobals->iOldY - y)/fDelta;
		if(abs(fValX)>abs(fValY))
			glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
		else
			glutSetCursor(GLUT_CURSOR_UP_DOWN);
		physicalglobals->V[selected_index] = glm::vec3(0);
		physicalglobals->P[selected_index].x += graphicalglobals->Right[0]*fValX;
		float fNewval = physicalglobals->P[selected_index].y + graphicalglobals->Up[1]*fValY;
		if(fNewval >0)
			physicalglobals->P[selected_index].y = fNewval;
		physicalglobals->P[selected_index].z += graphicalglobals->Right[2]*fValX + graphicalglobals->Up[2] * fValY;
	}
	graphicalglobals->iOldX = x;
	graphicalglobals->iOldY = y;

	glutPostRedisplay();
}
void keyboardFunc(unsigned char key, int x, int y)
{

}
void initilization()
{
	genMesh(10,4,4,0.1f,0.1f,0.1f);
	physicalglobals->iNumofTetrohedron = tetrahedra.size();

	physicalglobals->total_points = physicalglobals->P.size();
	physicalglobals->MASS.resize(physicalglobals->total_points);

	physicalglobals->A_row.resize(physicalglobals->total_points);
	physicalglobals->K_row.resize(physicalglobals->total_points);
	physicalglobals->b.resize(physicalglobals->total_points);
	physicalglobals->V.resize(physicalglobals->total_points);
	physicalglobals->F.resize(physicalglobals->total_points);
	physicalglobals->F0.resize(physicalglobals->total_points);
	cgglobals->residual.resize(physicalglobals->total_points);
	cgglobals->update.resize(physicalglobals->total_points);
	cgglobals->prev.resize(physicalglobals->total_points);

	memset(&(physicalglobals->V[0].x),0,physicalglobals->total_points*sizeof(glm::vec3));
	startTime = (float)glutGet(GLUT_ELAPSED_TIME);
	currentTime = startTime;

	QueryPerformanceFrequency(&frequency);

	QueryPerformanceCounter(&t1);
	
}