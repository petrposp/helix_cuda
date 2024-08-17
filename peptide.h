#ifndef PEPTIDE_H
#define PEPTIDE_H

#include <vector>
#include "residue.h"
#include <fstream>
#include <sstream>
#include <map>
#include <string>

using namespace std;

class Peptide {
public:
    Peptide() {}
    
    vector<Residue> seq;
    vector<string> population;
    int rot_angle, tilt_angle;
    double depth_h;
    double depth_b;
    double energy_h;
    double energy_b;
    double dG;
    string name;
    vector<int> ids;
    double totalP_h, totalP_b;
    
    unsigned long size() {
        return seq.size();
    }

    void load_sequences(string input) {
        std::fstream fs(input, std::fstream::in );
        string c1;

        while ( !fs.eof() ) {// Lines in input, eof = end of file 
            fs >> c1;

            population.push_back(c1);
        }
    }

    void initial_pos(string input, Peptide resids, map<char, int> res_id) // sequence manipulation
    {
        std::fstream init(input, std::fstream::in);
        double xpos, ypos, zpos;

        int i = 0;
        while (!init.eof() && i < 30) // Lines in input, For (steps)
        {
            init >> xpos >> ypos >> zpos;

            seq[i].pos.x = xpos;
            seq[i].pos.y = ypos;
            seq[i].pos.z = zpos;
	    //cout << zpos << endl;
            i++;
        }

        Atom cm=com();
        //
        // Move each bead so that it corresponds to the COM of the side chain and not the c-alfa
        // Maintain x position, but add y, z coordinates according to the position of com of the aa (calculated from com_aa)
        //

        for (int j=0; j<seq.size(); j++) {
            seq[j].pos = seq[j].pos-cm;

            Atom rest;
            rest.y = resids.seq[res_id[name[j]]].com.y; //-seq[j].pos.y;
            rest.z = resids.seq[res_id[name[j]]].com.z; //-seq[j].pos.z;

            seq[j].pos.y = seq[j].pos.y*(1+rest.y);
            seq[j].pos.z = seq[j].pos.z*(1+rest.z);
        }
    }

        void linear_pos() // sequence manipulation
    {
        seq[0].pos.x=0;
        seq[0].pos.y=0;
        seq[0].pos.z=0;
        for (int i=1;i<seq.size();++i) {
            seq[i].pos.x=seq[i-1].pos.x + 0.2;
            seq[i].pos.y=seq[0].pos.y;
            seq[i].pos.z=seq[0].pos.z;
        }
    }

    void com_aa(string input, map<char, int> res_id) // COM of aa, later used in initial_pos
    {
        std::fstream init( input, std::fstream::in);
        char aa;
        string type;
        int n;
        double xpos, ypos, zpos;

        int i=0;
        char aa_prev = 'A';
        int count=0;
        Atom cm;
        Atom COM=com();
        while (!init.eof() && i < 296) {

            init >> aa >> type >> n >> xpos >> ypos >> zpos;
            if(type == "CA") {
                seq[res_id[aa]].CA.x = xpos;
                seq[res_id[aa]].CA.y = ypos;
                seq[res_id[aa]].CA.z = zpos;
            }

            if (aa == aa_prev) {
                cm.x += xpos;
                cm.y += ypos;
                cm.z += zpos;
                ++count;

                cm *= 1.0/count;
            }
            else {
                count = 0;
                cm.x = 0;
                cm.y = 0;
                cm.z = 0;
            }

            seq[res_id[aa]].com.x=cm.x-COM.x;
            seq[res_id[aa]].com.y = cm.y-COM.y;
            seq[res_id[aa]].com.z = cm.z-COM.z;
            // asi  com sidechainu - COM celé aa

            aa_prev = aa;

            ++i;
        }
        // Move each bead so that it corresponds to the COM of the side chain and not the calfa
    }

    Atom com() {
        Atom cm;
        cm.x = 0;
        cm.y = 0;
        cm.z = 0;
        int count=0;
        for(auto& res : seq) {
            cm.x += res.pos.x;
            cm.y += res.pos.y;
            cm.z += res.pos.y;
            ++count;
        }
        cm *= 1.0/count; // dělení součtu atomů jejich počtem
        return cm;
    }

};
#endif //PEPTIDE_H
