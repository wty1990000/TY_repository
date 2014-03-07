#include "vgl.h"
#include <vmath.h>
#include "LoadShaders.h"
#include <iostream>

using namespace std;

enum VAO_IDS {Triangles, NumVAOs};
enum Buffer_IDS {ArrayBuffer, NumBuffers};
enum Arrib_IDS {vPosition=0, vColor=1};

const GLuint NumVertices = 6;

typedef struct global_t
{
	GLuint VAOs[NumVAOs];
	GLuint Buffers[NumBuffers];
}global;

global *glb;

void init(void)
{
	glEnable(GL_MULTISAMPLE);
	glb = new global();
	glGenBuffers(NumVAOs, glb->VAOs);
	glBindVertexArray(glb->VAOs[Triangles]);

	struct VertexData
	{
		GLubyte color[4];
		GLfloat position[4];
	};
	
	VertexData vertices[NumVertices] = {
		{{255,0,0,255}, {-0.90f, -0.90f}},
		{{0,255,0,255}, {0.85f, -0.90f}},
		{{0,0,255,255}, {-0.90f, 0.85f}},
		{{10,10,10,255}, {0.90f, -0.85f}},
		{{100,100,100,255}, {0.90f, 0.90f}},
		{{255,255,255,255}, {-0.85f, 0.90f}}
	};
	glGenBuffers(NumBuffers, glb->Buffers);
	glBindBuffer(GL_ARRAY_BUFFER, glb->Buffers[ArrayBuffer]);
	glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);

	ShaderInfo shaders[] = 
	{
		{GL_VERTEX_SHADER, "gouraud.vert"},
		{GL_FRAGMENT_SHADER, "gouraud.frag"},
		{GL_NONE, NULL}
	};

	GLuint program = LoadShaders(shaders);
	glUseProgram(program);

	glVertexAttribPointer(vColor, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(VertexData), BUFFER_OFFSET(0));
	glVertexAttribPointer(vPosition, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), BUFFER_OFFSET(sizeof(vertices[0].color)));

	glEnableVertexAttribArray(vColor);
	glEnableVertexAttribArray(vPosition);

	
}
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glBindVertexArray(glb->VAOs[Triangles]);

	glDrawArrays(GL_TRIANGLES, 0, NumVertices);
	
	glFlush();
}
int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowPosition(140, 140);
	glutInitWindowSize(1024, 768);
	glutCreateWindow(argv[0]);
	glutInitContextVersion(4,0);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	
	glewExperimental = true;
	if(glewInit()){
		cerr<<"Unable to initialize GLEW...exiting"<<endl;
		exit(EXIT_FAILURE);
	}
	init();
	glutDisplayFunc(display);

	glutMainLoop();

	return 0;
}