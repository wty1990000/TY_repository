#ifndef CONJUGATE_GRADIENT_H_
#define CONJUGATE_GRADIENT_H_

#include <glm/glm.hpp>
#include "utilities.h"

struct ConjugateGradient
{
	ConjugateGradient(){}

	std::vector<glm::vec3> residual;
	std::vector<glm::vec3> prev;
	std::vector<glm::vec3> update;

	const float tiny = 1e-010f;
	const float tolerence = 0.001f;
	const int i_max = 20;	
};

ConjugateGradient *cgglobals;

void initialize_CGsolver();
void conjugate_gradient_solver(const float &dt, ConjugateGradient &CG);

#endif // !CONJUGATE_GRADIENT_H_
