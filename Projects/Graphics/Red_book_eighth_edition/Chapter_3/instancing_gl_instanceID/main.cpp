#include "vgl.h"
#include <vmath.h>
#include "vutils.h"
#include "vbm.h"

#include <iostream>

#define INSTANCE_COUNT 100

using namespace std;
using namespace vmath;

typedef struct global_t
{
	float aspect;
	GLuint color_buffer;
	GLuint model_matrix_buffer;
	GLuint color_tbo;
	GLuint model_matrix_tbo;
	GLuint render_prog;

	GLuint view_matrix_loc;
	GLuint projection_matrix_loc;

	VBObject object;

}global;

global *glb;

void init()
{
	glb = new global();

	//Create the program for rendering the shader
	glb->render_prog = glCreateProgram();

	//Vertex Shader
	static const char shader_vs [] = 
		 "#version 410\n"
        "\n"
        "// 'position' and 'normal' are regular vertex attributes\n"
        "layout (location = 0) in vec4 position;\n"
        "layout (location = 1) in vec3 normal;\n"
        "\n"
        "// Color is a per-instance attribute\n"
        "layout (location = 2) in vec4 color;\n"
        "\n"
        "// The view matrix and the projection matrix are constant across a draw\n"
        "uniform mat4 view_matrix;\n"
        "uniform mat4 projection_matrix;\n"
        "\n"
        "// These are the TBOs that hold per-instance colors and per-instance\n"
        "// model matrices\n"
        "uniform samplerBuffer color_tbo;\n"
        "uniform samplerBuffer model_matrix_tbo;\n"
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
        "    // Use gl_InstanceID to obtain the instance color from the color TBO\n"
        "    vec4 color = texelFetch(color_tbo, gl_InstanceID);\n"
        "\n"
        "    // Generating the model matrix is more complex because you can't\n"
        "    // store mat4 data in a TBO. Instead, we need to store each matrix\n"
        "    // as four vec4 variables and assemble the matrix in the shader.\n"
        "    // First, fetch the four columns of the matrix (remember, matrices are\n"
        "    // stored in memory in column-primary order).\n"
        "    vec4 col1 = texelFetch(model_matrix_tbo, gl_InstanceID * 4);\n"
        "    vec4 col2 = texelFetch(model_matrix_tbo, gl_InstanceID * 4 + 1);\n"
        "    vec4 col3 = texelFetch(model_matrix_tbo, gl_InstanceID * 4 + 2);\n"
        "    vec4 col4 = texelFetch(model_matrix_tbo, gl_InstanceID * 4 + 3);\n"
        "\n"
        "    // Now assemble the four columns into a matrix.\n"
        "    mat4 model_matrix = mat4(col1, col2, col3, col4);\n"
        "\n"
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
	static const char shader_fs [] = 
		"#version 410\n"
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
	vglAttachShaderSource(glb->render_prog, GL_VERTEX_SHADER, shader_vs);
	vglAttachShaderSource(glb->render_prog, GL_FRAGMENT_SHADER, shader_fs);

	glLinkProgram(glb->render_prog);
	glUseProgram(glb->render_prog);

	glb->view_matrix_loc = glGetUniformLocation(glb->render_prog,"view_matrix");
	glb->projection_matrix_loc = glGetUniformLocation(glb->render_prog, "projection_matrix");

	GLuint color_tbo_loc = glGetUniformLocation(glb->render_prog, "color_tbo");
	GLuint model_matrix_tbo_loc = glGetUniformLocation(glb->render_prog, "model_matrix_tbo");

	//Unit texture index
	glUniform1i(color_tbo_loc,0);
	glUniform1i(model_matrix_tbo_loc, 1);

	glb->object.LoadFromVBM("armadillo_low.vbm",0,1,2);

	//Set up th texture buffers
	glGenTextures(1, &glb->color_tbo);
	glBindTexture(GL_TEXTURE_BUFFER, glb->color_buffer);

	vec4 colors[INSTANCE_COUNT];
	for(int n =0; n< INSTANCE_COUNT; n++){
		float a = float(n) /4.0f;
		float b = float(n) /5.0f;
		float c = float(n) /6.0f;

		colors[n][0] = 0.5f + 0.25f * (sinf(a + 1.0f) + 1.0f);
        colors[n][1] = 0.5f + 0.25f * (sinf(b + 2.0f) + 1.0f);
        colors[n][2] = 0.5f + 0.25f * (sinf(c + 3.0f) + 1.0f);
        colors[n][3] = 1.0f;
	}

	glGenBuffers(1, &glb->color_buffer);
	glBindBuffer(GL_TEXTURE_BUFFER,glb->color_buffer);
	glBufferData(GL_TEXTURE_BUFFER,sizeof(colors),colors,GL_STATIC_DRAW);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, glb->color_buffer);

	glGenTextures(1, &glb->model_matrix_tbo);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER, glb->model_matrix_tbo);
	glGenBuffers(1, &glb->model_matrix_buffer);
	glBindBuffer(GL_TEXTURE_BUFFER, glb->model_matrix_buffer);
	glBufferData(GL_TEXTURE_BUFFER, INSTANCE_COUNT * sizeof(mat4), NULL, GL_DYNAMIC_DRAW);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, glb->model_matrix_buffer);
	glActiveTexture(GL_TEXTURE0);
}
void display(void)
{
	float t = float(GetTickCount() & 0x3FFF )/ float(0x3FFF);
	static const vec3 X(1.0f, 0.0f, 0.0f);
    static const vec3 Y(0.0f, 1.0f, 0.0f);
    static const vec3 Z(0.0f, 0.0f, 1.0f);

	mat4 matrices[INSTANCE_COUNT];
	
	for(int n=0; n< INSTANCE_COUNT; n++){
		 float a = 50.0f * float(n) / 4.0f;
        float b = 50.0f * float(n) / 5.0f;
        float c = 50.0f * float(n) / 6.0f;

        matrices[n] = rotate(a + t * 360.0f, 1.0f, 0.0f, 0.0f) *
                      rotate(b + t * 360.0f, 0.0f, 1.0f, 0.0f) *
                      rotate(c + t * 360.0f, 0.0f, 0.0f, 1.0f) *
                      translate(10.0f + a, 40.0f + b, 50.0f + c);
	}
	glBindBuffer(GL_TEXTURE_BUFFER, glb->model_matrix_buffer);
	glBufferData(GL_TEXTURE_BUFFER, sizeof(matrices), matrices, GL_DYNAMIC_DRAW);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glUseProgram(glb->render_prog);

	mat4 view_matrix(translate(0.0f, 0.0f, -1500.0f) * rotate(t * 360.0f * 2.0f, 0.0f, 1.0f, 0.0f));
	mat4 projection_matrix(frustum(-1.0f, 1.0f, -glb->aspect, glb->aspect, 1.0f, 5000.0f));

	glUniformMatrix4fv(glb->view_matrix_loc, 1, GL_FALSE, view_matrix);
	glUniformMatrix4fv(glb->projection_matrix_loc, 1, GL_FALSE, projection_matrix);

	glb->object.Render(0, INSTANCE_COUNT);

	glFlush();
	glutSwapBuffers();
	glutPostRedisplay();
}

void close(void)
{
	glUseProgram(0);
	glDeleteProgram(glb->render_prog);
	glDeleteBuffers(1, &glb->color_buffer);
	glDeleteBuffers(1, &glb->model_matrix_buffer);
	delete glb;
}

void reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	glb->aspect = float(width) / float(height);
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
	glutInitContextVersion(4,1);
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