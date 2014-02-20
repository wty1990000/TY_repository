#ifndef _PHYSICALSIMULATION_H_
#define _PHYSICALSIMULATION_H_

#include <glm/glm.hpp>
#include "TetMesh.h"
#include "utilities.h"

typedef std::map<int, glm::mat3> matrix_map;
typedef matrix_map::iterator matrix_iterator;

struct PhysicalGlobalVariables
{
	int iNumofTetrohedron;
	size_t total_points;
	std::vector<glm::vec3> Xi;	//Material coordinates
	std::vector<glm::vec3> P;	//world coordinates
	std::vector<glm::vec3> V;	//velocity
	std::vector<float>	MASS;	//mass matrix
	std::vector<glm::vec3> F;	//Forces
	std::vector<bool> IsFixed;		//fixed points
	glm::mat3 eye;
	
	std::vector<matrix_map> K_row;
	std::vector<matrix_iterator> A_row;
	std::vector<glm::vec3> F0;
	std::vector<glm::vec3> b;

	bool bUsingStiffnessWarping;

	PhysicalGlobalVariables()
		:iNumofTetrohedron(0),total_points(0),eye(glm::vec3(1)),bUsingStiffnessWarping(true)
	{}
};

PhysicalGlobalVariables *physicalglobals;

/* --------- Physical Quantities -----------*/
const glm::vec3 gravAcceleration = glm::vec3(0.0f, -9.81f, 0.0f);	//Gravitational acceleration	
const float fDensity = 1000.0f;										//mass density
const float fDamping = 1.0f;
const float fPoisonRatio = 0.33f;			//Poisson ratio
const float fYoungModulus = 50000.0f;		//Young's modulus
const int creep = 0.20f;
const int yield = 0.04f;
const float m_max = 0.2f;

/* --------- matrix parameters used in Hookean material constitutive model -----------*/
const float fGeneralMultiplier = fYoungModulus / (1.0f + fPoisonRatio) / (1.0f - 2 * fPoisonRatio);	//d15
const float fType1 = (1.0f - fPoisonRatio) * fGeneralMultiplier;									//d16
const float fType2 = fPoisonRatio * fGeneralMultiplier;												//d17
const float fType3 = (1.0f - 2.0f * fPoisonRatio) * fGeneralMultiplier;								//d18
const glm::vec3 hookLinearElastisity(fType1, fType2, fType3);										//Matrix D isotropic elasticity

/* --------- functions for physical computing -----------*/
void initializePhysics();
float GetTetrahedronVolume(glm::vec3 e1, glm::vec3 e2, glm::vec3 e3);
void addTetraheron(int i0, int i1, int i2, int i3);
void genMesh(size_t xdim, size_t ydim, size_t zdim, float fWidth, float fHeight, float fDepth);
void computeforce();
void recalcmassmatrix();

#endif // !_PHYSICALSIMULATION_H_
