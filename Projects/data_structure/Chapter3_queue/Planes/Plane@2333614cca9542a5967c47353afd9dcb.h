#ifndef PLANE_H_
#define PLANE_H_

enum Plane_status
{
	null, arriving, departing;
};

class Plane{
public:
	Plane();
	Plane(inti flt, int time, Plane_status status);
	void refuse() const;
	void land(int time) const;
	void fly(int time) const;
	int started() const;
};
#endif // !PLANE_H_
