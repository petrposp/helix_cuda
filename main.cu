#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <sstream>
#include <ostream>
#include <chrono>
#include <map>
#include <algorithm>
#include <chrono>

#include "peptide.h"
#include "energy_calculator.h"

#define NUMBER_PEPTIDES 15 //20000
using namespace std;

const double radToDeg = 57.2958;
const double degToRad = 1.0/57.2958;

int main() {
    auto start = chrono::high_resolution_clock::now();
    Peptide pep;
    Energy_calculator calc;

    Peptide resids;
    resids.name = "AEKWRCVLIMQFNDSTY12345678900";
    resids.seq.resize(resids.name.size());
    map<char, int> res_id { { 'A', 0 }, { 'E', 1 }, { 'K', 2 },  { 'W', 3 }, { 'R', 4 }, { 'C', 5 }, { 'V', 6 }, { 'L', 7 }, { 'I', 8 },  { 'M', 9 }, { 'Q', 10 }, { 'F', 11 },  { 'N', 12 }, {'D', 13}, {'S', 14}, {'T', 15}, {'Y', 16},{'1',17},{'2',18},{'3',19},{'4',20},{'5',21},{'6',22},{'7',23},{'8',24},{'9',25},{'0',26},{'O',27}};

    calc.init_grid(resids.name.size(), 200, 2);

    //Load energy files into resids.seq.E

    for (int j=0; j < resids.name.size(); j++) {
        calc.load_EfilesH(resids.name[j], j);
        calc.load_EfilesB(resids.name[j], j);
        //resids.seq[j].init(calc.E_H[res_id[resids.name[j]]], 1);
        resids.seq[j].init(calc.E_H[j], 1);
        //resids.seq[j].init(calc.E_B[res_id[resids.name[j]]], 2);
    	resids.seq[j].init(calc.E_B[j], 2);
    }
    // Energies to arrays
    double *E_H_line;
    double *E_B_line;

    cudaMallocManaged(&E_H_line, 28*200*2*sizeof(*E_H_line));
    cudaMallocManaged(&E_B_line, 28*200*2*sizeof(*E_B_line));

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 200; j++) {
            E_H_line[i*200*2+j*2] = calc.E_H[i][j][0];
	        E_H_line[i*200*2+j*2+1] = calc.E_H[i][j][1];
	        E_B_line[i*200*2+j*2] = calc.E_B[i][j][0];
	        E_B_line[i*200*2+j*2+1] = calc.E_B[i][j][1];
            }
    }

    //exit(0);
    // Load charged residues profile --> average between charged and neutral forms

    //load_charged(calc, resids, res_id, 1);
    //load_charged(calc, resids, res_id, 2);

    // Load COM of each amino acid
    resids.com_aa("all.gro", res_id);

    pep.load_sequences("input");

	// pro více peptidů
	double *positions_z;
	int *ids;
	cudaMallocManaged(&positions_z, NUMBER_PEPTIDES*30*sizeof(*positions_z));
	cudaMallocManaged(&ids, NUMBER_PEPTIDES*30*sizeof(*ids));

	double min[NUMBER_PEPTIDES] = {999999}, minB[NUMBER_PEPTIDES] = {999999};
	double depth_h[NUMBER_PEPTIDES] = {0}, depth_b[NUMBER_PEPTIDES] = {0};

    for (int a=0; a < pep.population.size(); a++) {
        pep.name = pep.population[a]; // populace je celý input, jeden name je jedna linie
	    pep.ids.resize(30);
	    for (int n= 0; n < pep.name.size(); n++) {
		    pep.ids[n] = res_id[pep.name[n]];
	    }
        pep.seq.resize(30); //délka sekvence má být 20 Vector residuí

        pep.initial_pos("init_pos30", resids, res_id); // resids obsahuje všechny energie pro každou aminokyselinu

    	// get energy musí vrátit pole pro energies_h/b, total_B_en_h/b, 
        // calc.get_energy_total(pep, resids, res_id, E_H_line, E_B_line); // pep je aktuální peptid, resids všechny energie, res_id je mapa pro propojení aminokyseliny s číslem

	    calc.prepare(pep, resids, a % NUMBER_PEPTIDES, positions_z, ids);
	    // pro každý peptid individuální pozice

    	if (((a + 1) % NUMBER_PEPTIDES) == 0) {
            calc.get_energy_total_multiple(positions_z, ids, E_H_line, E_B_line, min, minB, depth_h, depth_b);

            for (int i = NUMBER_PEPTIDES; i > 0; i--) {
                //if (min[NUMBER_PEPTIDES - i] - minB[NUMBER_PEPTIDES - i] > 10) {
                    cout << pep.population[(a+1) - i] << " " << min[NUMBER_PEPTIDES-i] << " " << minB[NUMBER_PEPTIDES-i] << " " <<  min[NUMBER_PEPTIDES-i]-minB[NUMBER_PEPTIDES-i] << " " << depth_h[NUMBER_PEPTIDES-i] << " " << depth_b[NUMBER_PEPTIDES-i] << endl;
                //}
            }
        }
        //cout << pep.name << " " << pep.energy_h << " " << pep.energy_b << " " << pep.energy_h-pep.energy_b << " " << pep.depth_h << " " << pep.depth_b << endl;

    }
}
