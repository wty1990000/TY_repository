#include "vgl.h"
#include <vmath.h>
#include "vbm.h"
#include "vutils.h"

#include "LoadShaders.h"
#include <iostream>

using namespace std;
using namespace vmath;

#define INSTANCE_COUNT 100

typedef struct instance_t
{
	float aspect;

	GLuint color_buffer;
	GLuint model_matrix_buffer;
	GLuint render_prog;
	GLuint moder_matrix_loc;
	GLuint view_matrix_loc;
	GLuint projection_matrix_loc;

	VBObject object;
}global;

global *glb;

void init()
{
	glb = new global();

	glb->render_prog = glCreateProgram();

	//Vertex shader
	static const char shader_VS[] = 
		"#version 330\n"
        "\n"
        "// 'position' and 'normal' are regular vertex attributes\n"
        "layout (location = 0) in vec4 position;\n"
        "layout (location = 1) in vec3 normal;\n"
        "\n"
        "// Color is a per-instance attribute\n"
        "layout (location = 2) in vec4 color;\n"
        "\n"
        "// model_matrix will be used as a per-instance transformation\n"
        "// matrix. Note that a mat4 consumes 4 consecutive locations, so\n"
        "// this will actually sit in locations, 3, 4, 5, and 6.\n"
        "layout (location = 3) in mat4 model_matrix;\n"
        "\n"
        "// The view matrix and the projection matrix are constant across a draw\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "\n"
        "// The output of the vertex shader (matched to the fragment shader)\n"
        "out VERTEX\n"
        "{\n"
        "    vec3    normal;\n"
        "    vec4    color;\n"
        "} vertex;\n"
        "\n"
        "// Ok, go!\n"
        "void main(void)\n"
        "{\n"
        "    // Construct a model-view matrix from the uniform view matrix\n"
        "    // and the per-instance model matrix.\n"
        "    mat4 model_view_matrix = view_matrix * model_matrix;\n"
        "\n"
        "    // Transform position by the model-view matrix, then by the\n"
        "    // projection matrix.\n"
        "    gl_Position = projection_matrix * (model_view_matrix * position);\n"
        "    // Transform the normal by the upper-left-3x3-submatrix of the\n"
        "    // model-view matrix\n"
        "    vertex.normal = mat3(model_view_matrix) * normal;\n"
        "    // Pass the per-instance color through to the fragment shader.\n"
        "    vertex.color = color;\n"
        "}\n";

	//Fragment shader
	static const char shader_FS [] =
		"#version 330\n"
        "\n"
        "layout (location = 0) out vec4 color;\n"
        "\n"
        "in VERTEX\n"
        "{\n"
        "    vec3    normal;\n"
        "    vec4    color;\n"
        "} vertex;\n"
        "\n"
        "void main(void)\n"
        "{\n"
        "    color = vertex.color * (0.1 + abs(vertex.normal.z)) + vec4(0.8, 0.9, 0.7, 1.0) * pow(abs(vertex.normal.z), 40.0);\n"
        "}\n";

	vglAttachShaderSource(glb->render_prog, GL_VERTEX_SHADER, shader_VS);
	vglAttachShaderSource(glb->render_prog, GL_FRAGMENT_SHADER, shader_FS);

	glLinkProgram(glb->render_prog);
	glUseProgram(glb->render_prog);

	//get the locations
	glb->view_matrix_loc = glGetUniformLocation(glb->render_prog,"view_matrix");
	glb->projection_matrix_loc = glGetUniformLocation(glb->render_prog, "projection_matrix");

	//Load object
	glb->object.LoadFromVBM("armadillo_low.vbm", 0, 1, 2);
	//Bind the object
	glb->object.BindVertexArray();

	//Get locations
	int position_loc = glGetAttribLocation(glb->render_prog, "position");
	int normal_loc	 = glGetAttribLocation(glb->render_prog, "normal");
	int color_loc	 = glGetAttribLocation(glb->render_prog, "color");
	int matrix_loc   = glGetAttribLocation(glb->render_prog, "model_matrix");

	vec4 colors[INSTANCE_COUNT];

	for(int n=0; n< INSTANCE_COUNT; n++){
		float a = float(n) / 4.0f;
		float b = float(n) / 5.0f;
		float c = float(n) / 6.0f;

		colors[n][0] = 0.5f + 0.25f * (sinf(a+1.0f) + 1.0f);
		colors[n][1] = 0.5f + 0.25f * (sinf(b+2.0f) + 1.0f);
		colors[n][2] = 0.5f + 0.25f * (sinf(c+3.0f) + 1.0f);
		colors[n][3] = 1.0f;
	}

	glGenBuffers(1, &glb->color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, glb->color_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_DYNAMIC_DRAW);

	//Set up color array
	glBindBuffer(GL_ARRAY_BUFFER, glb->color_buffer);
	glVertexAttribPointer(color_loc, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(color_loc);

	//change color per-instance
	glVertexAttribDivisor(color_loc,1);

	//Do the same things for the model matrix
	glGenBuffers(1, &glb->model_matrix_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, glb->model_matrix_buffer);
	glBufferData(GL_ARRAY_BUFFER, INSTANCE_COUNT * sizeof(mat4), NULL, GL_DYNAMIC_DRAW);
	for(int i = 0; i< 4; i++){
		glVertexAttribPointer(matrix_loc+i, 4, GL_FLOAT, GL_FALSE, sizeof(mat4),BUFFER_OFFSET(sizeof(vec4)*i));
		glEnableVertexAttribArray(matrix_loc+i);
		glVertexAttribDivisor(matrix_loc+i, 1);
	}

	//Done. Unbind the VBO
	glBindVertexArray(0);
}

static inline int MINIMUM(int a, int b)
{
	return a<b? a: b;
}

void display(void)
{
	float t = float(GetTickCount() & 0x3FFF) / float(0x3FFF);
	static const vec3 X(1.0f, 0.0f, 0.0f);
	static const vec3 Y(0.0f, 1.0f, 0.0f);
	static const vec3 Z(0.0f, 0.0f, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

	glBindBuffer(GL_ARRAY_BUFFER, glb->model_matrix_buffer);

	mat4* matrices = (mat4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for(int n =0; n < INSTANCE_COUNT; n++){
		float a = 50.0f * float(n) / 4.0f;
		float b = 50.0f * float(n) / 5.0f;
		float c = 50.0f * float(n) / 6.0f;

		matrices[n] = rotate(a + t * 360.0f, 1.0f, 0.0f, 1.0f)*
					  rotate(b + t * 360.0f, 0.0f, 1.0f, 0.0f)*
					  rotate(c + t * 360.0f, 0.0f, 0.0f, 1.0f)*
					  translate(10.0f+a, 40.0f+b, 50.0f+c);
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glUseProgram(glb->render_prog);

	mat4 view_matrix(translate(0.0f, 0.0f, -1500.0f) * rotate(t * 360.f * 2.0f, 0.0f, 1.0f, 0.0f));
	mat4 projection_matrix(frustum(-1.0f, 1.0f, -glb->aspect, glb->aspect, 1.0f, 500.0f));

	glUniformMatrix4fv(glb->view_matrix_loc, 1, GL_FALSE, view_matrix);
	glUniformMatrix4fv(glb->projection_matrix_loc, 1, GL_FALSE, projection_matrix);

	glb->object.Render(0, INSTANCE_COUNT);

	glFlush();
	glutSwapBuffers();
	glutPostRedisplay();
}

void reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	glb->aspect = float(width) / float(height);
}

void close(void)
{
	glUseProgram(0);
	glDeleteProgram(glb->render_prog);
	glDeleteBuffers(1, &glb->color_buffer);
	glDeleteBuffers(1, &glb->model_matrix_buffer);
	delete glb;
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);
		break;
	default:
		break;
	}
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowPosition(140, 140);
	glutInitWindowSize(1024, 768);
	glutInitContextVersion(3,3);
	glutInitContextProfile(GLUT_CORE_PROFILE);
	glutCreateWindow(argv[0]);
	glewExperimental = GL_TRUE;
	if(glewInit()){
		cerr<<"Unable to initialize GLEW...exite"<<endl;
		exit(EXIT_FAILURE);
	}
	init();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutCloseFunc(close);
	glutMainLoop();
	
	return 0;
}