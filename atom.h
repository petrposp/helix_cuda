#ifndef ATOM_H
#define ATOM_H

#include <cmath>

class Atom{
public:
    double x, y, z;

    Atom() {}
    Atom(double x, double y, double z): x(x), y(y), z(z) {
    }

    // tvoří co se děje, když chci porovnat atom == atom2
    bool operator==(const Atom& o) const {
        if(x != o.x) return false;
        if(y != o.y) return false;
        if(z != o.z) return false;

        return true;
    }

    bool operator!=(const Atom& o) const {
        return !(*this == o);
    }

    Atom operator+(const Atom& o) const {
        return Atom(x+o.x, y+o.y, z+o.z);
    }

    void operator+=(const Atom& o) {
        x += o.x;
        y += o.y;
        z += o.z;
    }

    void operator-=(const Atom& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
    }

    Atom operator-(const Atom& o) const {
        return Atom(x-o.x, y-o.y, z-o.z);
    }

    Atom operator*(const double a) const {
        return Atom(x*a, y*a, z*a);
    }

    Atom operator/(const double a) const {
        return Atom(x/a, y/a, z/a);
    }
    void operator*=(double a) {
        x*=a;
        y*=a;
        z*=a;
    }

    inline void rotate(Atom& axis, double angle) {
        angle*=0.5;
        double cosAngle = cos(angle);
        double sinAngle = sin(angle);
        double t2, t3, t4, t5, t6, t7, t8, t9, t10, newx, newy, newz;
        double qw = cosAngle, qx (axis.x * sinAngle), qy = (axis.y * sinAngle), qz = (axis.z * sinAngle);
        
        t2 =  qw * qx;
        t3 =  qw * qy;
        t4 =  qw * qz;
        t5 = -qx * qx;
        t6 =  qx * qy;
        t7 =  qx * qz;
        t8 = -qy * qy;
        t9 =  qy * qz;
        t10 = -qz * qz;

        newx = 2.0 * ( (t8+t10) * x + (t6-t4)  * y + (t3+t7) * z ) + x;
        newy = 2.0 * ( (t4+t6)  * x + (t5+t10) * y + (t9-t2) * z ) + y;
        newz = 2.0 * ( (t7-t3)  * x + (t2+t9)  * y + (t5+t8) * z ) + z;

        x = newx;
        y = newy;
        z = newz;
    }

    double size() const {
        return sqrt(x*x + y*y + z*z);
    }

    inline void normalise() {
        double tot = size();
        if (tot != 0.0) {
            tot = 1.0/tot;
            x *= tot;
            y *= tot;
            z *= tot;
        }
    }
};

#endif // ATOM_H
