#include <glm/glm.hpp>
#include "TetMesh.h"

/* --------- matrix parameters used in Hookean material constitutive model -----------*/
const float fPoisonRatio = 0.33f;			//Poisson ratio
const float fYoungModulus = 50000.0f;		//Young's modulus
const float fDensity = 1000.0f;
const float fGeneralMultiplier = fYoungModulus / (1.0f + fPoisonRatio) / (1.0f - 2 * fPoisonRatio);	//d15
const float fType1 = (1.0f - fPoisonRatio) * fGeneralMultiplier;									//d16
const float fType2 = fPoisonRatio * fGeneralMultiplier;												//d17
const float fType3 = (1.0f - 2.0f * fPoisonRatio) * fGeneralMultiplier;								//d18
const glm::vec3 hookLinearElastisity(fType1, fType2, fType3);										//Matrix D isotropic elasticity