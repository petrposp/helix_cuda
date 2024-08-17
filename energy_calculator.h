#ifndef ENERGY_CALCULATOR_H
#define ENERGY_CALCULATOR_H

#include <vector>
#include "atom.h"
#include "peptide.h"
#include "residue.h"
#include "chrono"
#include <sys/time.h>
#include "map"
#include <string>

#define NUMBER_PEPTIDES 15
#define DEPTH 40
#define SEQ 30
using namespace std;

__device__ double get_E_human(double z, int inde, double *E_H) {
	double lower_bound = E_H[inde*200*2 + 199*2];
	double upper_bound = E_H[inde*200*2];
	double factor_Eh = (upper_bound - lower_bound)/199;
	double inv_factor_Eh = 1.0/factor_Eh;

	if (z < lower_bound) {
		z=z*-1;
	}
	if (z >= upper_bound) {
		return 0.0;
	}

	int convert = int((upper_bound-z) * inv_factor_Eh);

	double proportional_remainder = (z*inv_factor_Eh) - int(z*inv_factor_Eh);

	if (convert >= 198) {
		return (E_H[inde*200*2 + 199*2 + 1]);
	}
	else {
		return (1.0 - proportional_remainder)*E_H[inde*200*2 + convert*2 + 1] + proportional_remainder*(E_H[inde*200*2 + (convert+1)*2 + 1]);
	}
}

__device__ double get_E_bacteria(double z, int inde, double *E_B) {
        double lower_bound = E_B[inde*200*2 + 199*2];
        double upper_bound = E_B[inde*200*2];
        double factor_Eb = (upper_bound - lower_bound)/199;
        double inv_factor_Eb = 1.0/factor_Eb;

	if(z < lower_bound) {
            z=z*-1;
        }

        if(z >= upper_bound) {
            return 0.0;
        }

        int convert = int ((upper_bound-z) * inv_factor_Eb);

        double proportional_remainder = (z * inv_factor_Eb) - int(z *inv_factor_Eb);

        if(convert >= 198){
            return (E_B[inde*200*2 + 199*2 + 1]);
        }else{
            return  (1.0-proportional_remainder)*E_B[inde*200*2 + convert*2 + 1] + proportional_remainder*E_B[inde*200*2 + (convert+1)*2 + 1];

        }
    }


__global__ void get_energy(double *ret, int depth, double *energies_h, double *energies_b, double *positions_z, int *ids, double *total_B_en_h, double *total_B_en_b, double *E_H, double *E_B) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int local_x = threadIdx.x; // hloubka/2
    int local_y = threadIdx.y; // aminokyselina

	double cte = 1.0/(0.008314*310);
	double cte2 = 0.008314*310;
	double e = 2.718281828459045;
	double total_en_h = 0;
    double total_en_b = 0;
    
	__shared__ double en_h[20*33];
	__shared__ double en_b[20*33];

    // shaha inicializovat nepoužité 3 poslední sloupce na nuly
	/*if (local_y == 0) {
		for (int i =30; i < 33; i++) {
			en_b[x*33 + i] = 0;
			en_h[x*33 + i] = 0;
		}
	}*/

	/*if (local_y < 3) {
		en_h[x*33 + 30 + local_y] = 0;
		en_b[x*33 + 30 + local_y] = 0;
	}*/
	__syncthreads();

	double B_en_h = 0;
	double B_en_b = 0;

    double disp = x*0.1;

    double position_z = positions_z[blockIdx.y*30 + local_y] + disp;

    int inde = ids[blockIdx.y*30 + local_y];

    //uložit energii do shared, aby se mohlo sečíst
	en_h[local_x*33 + local_y] = get_E_human(position_z, inde, E_H);
	en_b[local_x*33 + local_y] = get_E_bacteria(position_z, inde, E_B);
	__syncthreads();

    // kontrola vypočtených energií
	// if (blockIdx.y == 14) {
	// 	ret[x*33 + local_y]  = en_b[local_x*33 + local_y];
	// 	ret[x*33 + 30] = en_b[local_x*33 + 30];
	// 	ret[x*33 + 31] = en_b[local_x*33 + 31];
	// 	ret[x*33 + 32] = en_b[local_x*33 + 32];

	// }

	// parallel reduction
	for (unsigned int s = 32/2; s > 0; s >>= 1) { // posouvá se bitový operátor o jedno doprava, místo 30 vláken, 32
		if (local_y < s) {
			en_h[local_x*33 + local_y] += en_h[local_x*33 + local_y + s];
			en_b[local_x*33 + local_y] += en_b[local_x*33 + local_y + s];
		}
		__syncthreads();
	}

	//shuffle reduction nepovedlo se mi přizpůsobit na sčítání napříč řádky
	/*for (int mask = warpSize/2; mask > 0; mask >>= 1) {
	    en_h[local_x*32 + local_y] += __shfl_down_sync(0xffffffff, en_h[local_x*32+local_y], mask);
	    en_b[local_x*32 + local_y] += __shfl_down_sync(0xffffffff, en_b[local_x*32 + local_y], mask);
	}*/

	if (local_y == 0) {
		total_en_h = en_h[local_x*33];
		total_en_b = en_b[local_x*33];
		//ret[blockIdx.y*40 + blockIdx.x*20 + local_x] = total_en_b;

		B_en_h = pow(e, -total_en_h*cte);
		atomicAdd(&total_B_en_h[blockIdx.y], B_en_h);
		energies_h[blockIdx.y*40 + x] = B_en_h;
		//ret[blockIdx.y*40 + x] = B_en_h;

		B_en_b = pow(e, -total_en_b*cte);
		atomicAdd(&total_B_en_b[blockIdx.y], B_en_b);
		energies_b[blockIdx.y*40 + x] = B_en_b;
	}

	return;
}




class Ener {
public:
    double energy;
    double tilt;
};

class Energy_calculator {
private:
    const double radToDeg = 57.2958;
    const double degToRad = 1.0/57.2958;
// v konstruktoru malloc a nachystat ukazatele, místo v get_energy_multiple

public:
    Energy_calculator() {}
    vector<double> test;

    int rot_deg = 360;
    int tilt_deg = 360;
    int depth = 40;
    vector< vector < vector < double >>> E_H;
    vector< vector < vector < double >>> E_B;
    vector< vector < vector < double >>> Charged;
    vector< vector < vector < double >>> Charged_B;
    double cte=1.0/(0.008314*310);
    double cte2=0.008314*310;
    double e=2.718281828459045;

    
    void load_EfilesH(char input, int n) {
        string in;
        in.push_back(input);
        string human = in + "h";
        std::fstream fs( human, std::fstream::in );
        double zz, ee;

        int i = 0;
        while (!fs.eof() && i<200)  {// Lines in input
            fs >> zz >> ee;
            E_H[n][i][0] = zz; //z poloha
            E_H[n][i][1] = ee; //energie pro polohu

            ++i;
        }
    }

    void load_EfilesB(char input, int n) {
        string in;
        in.push_back(input);
        string bact = in + "b";
        std::fstream fs( bact, std::fstream::in);
        double zz, ee;

        int i = 0;
        while (!fs.eof() && i < 200) {// Lines in input
            fs >> zz >> ee; //načítá vstup do proměnných
            E_B[n][i][0] = zz;
            E_B[n][i][1] = ee;
            
            ++i;
        }
    }

    void load_E_charged(string input, int n) {
        std::fstream fs(input, std::fstream::in);
        double zz, ee;
        int i=0;
        while (!fs.eof() && i < 200) {// Lines in input
            fs >> zz >> ee; //načítá vstup do proměnných
            Charged[n][i][0] = zz;
            Charged[n][i][1] = ee;
            
            ++i;
        }
    }
    
    void init_grid(int a, int b, int c) {
        E_H.resize(a, vector<vector<double>>(b, vector<double>(c)));
        for (int x=0; x<a; ++x) {
            for (int y=0; y<b; ++y) {
                for (int z=0; z<c; ++z) {
                    E_H[x][y][z] = 0;
                }
            }
        }
        E_B.resize(a, vector<vector<double>>(b, vector<double>(c)));
        for (int x=0; x<a; ++x) {
            for (int y=0; y<b; ++y) {
                for (int z=0; z<c; ++z) {
                    E_B[x][y][z] = 0;
                }
            }
        }
    }

    void prepare(Peptide& current, Peptide aa, int a, double *positions_z, int *ids) {
	for (int i=0; i < current.seq.size(); i++) {
		positions_z[a*30 + i] = current.seq[i].pos.z;
		ids[a*30 + i] = current.ids[i];
	}
	return;
    }

    // kalkulace pro x peptidů zároveň
    void get_energy_total_multiple(double *positions_z, int *ids, double *E_H, double *E_B, double *min, double *minB, double *depth_h, double *depth_b) {
	double G_all[40*NUMBER_PEPTIDES];
	double G_all_b[40*NUMBER_PEPTIDES];

	double *energies_h;
	double *energies_b;
	cudaMallocManaged(&energies_h, NUMBER_PEPTIDES*depth*sizeof(*energies_h));
    cudaMallocManaged(&energies_b, NUMBER_PEPTIDES*depth*sizeof(*energies_b));

    //double min=999999;
    //double minB = 999999;
    double *total_B_en_h;
    double *total_B_en_b;
    cudaMallocManaged(&total_B_en_h, NUMBER_PEPTIDES*sizeof(*total_B_en_h));
    cudaMallocManaged(&total_B_en_b, NUMBER_PEPTIDES*sizeof(*total_B_en_b));

    double *ret;
    //cudaMallocManaged(&ret, sizeof(*ret));
    cudaMallocManaged(&ret, 20*33*sizeof(*ret));

    double shift = 0;
    double shiftB = 0;

	dim3 Blocks(2, NUMBER_PEPTIDES);
	dim3 threadsPerBlock(depth/2, 30);
	get_energy<<< Blocks, threadsPerBlock >>>(ret, depth, energies_h, energies_b, positions_z, ids, total_B_en_h, total_B_en_b, E_H, E_B);
	cudaDeviceSynchronize();

/*	for (int a=0; a < 40; a++) {
		for (int b=0; b < 33; b++) {
			cout << ret[a*33 + b] << ", ";
		}
		cout << endl;
	}
*/
	for (int i = 0; i < NUMBER_PEPTIDES; i++) {
		min[i] = 999999;
		minB[i] = 999999;
		double G, Gb;
        	for (int q=0; q < depth; q++) {
	        	G = -cte2*log(energies_h[i*40 + q]/(total_B_en_h[i]));
		        G_all[i*40 + q] = G;
        		Gb = -cte2*log(energies_b[i*40 + q]/(total_B_en_b[i]));
            		G_all_b[i*40 + q] = Gb;
		}

		double z, zb;
		shift = G_all[(i+1)*depth-1];
		shiftB = G_all_b[(i+1)*depth-1];

		for (int r=0; r < depth; r++) {
			z = r*0.1;
			zb = r*0.1;
			if (G_all[i*40 + r] < min[i]) {
				min[i] = G_all[i*40 + r];
				depth_h[i] = z;
			}

			if (G_all_b[i*40 + r] < minB[i]) {
				minB[i] = G_all_b[i*40 + r];
				depth_b[i] = zb;
			}

		}
		min[i] -= shift;
		minB[i] -= shiftB;
	}
/*	cudaFree(G_all);
	cudaFree(G_all_b);
	cudaFree(energies_h);
	cudaFree(energies_b);
	cudaFree(total_B_en_h);
	cudaFree(total_B_en_b);*/
	//cudaFree(positions_z);
	//cudaFree(ids);
	return;
    }


    void print_Gplot(Peptide& current, Peptide aa, map<char, int> res_id, string name) {
        Atom axis_y = Atom(0, 1, 0);
        Atom axis_x = Atom(1, 0, 0);
        int winds=40; // 200-250
        double width = 1; //0.155
        vector<vector<double>> energies_h;
        energies_h.resize(winds, vector<double> (rot_deg*tilt_deg/5));
        vector<vector<double>> energies_b;
        energies_b.resize(winds, vector<double> (rot_deg*tilt_deg/5));
        vector<double> G_all;
        vector<double> G_allB;
        //double min=999999;
        double total_B_en = 0;
        double total_B_en_b = 0;
        double prob_sum = 0;
        double prob_sumB = 0;
        double shift = 0;
        double shiftB = 0;

        string fileH = name+ "_GH";
        string fileB= name + "_GB";

        double total_en_h = 0;
        double total_en_b = 0;

        for (int k=0; k < winds; k++) {
            double disp = k*0.1*width;

            for (int i = 0; i < rot_deg; i=i+5) {
                double deg_rot = i*degToRad;

                for (int j = 0; j < tilt_deg; j=j+5) {
                    double deg_tilt = j*degToRad;

                    total_en_h = 0;
                    total_en_b = 0;

                    // Move rotates and get E under peptide class
                    for (int a=0; a<current.seq.size(); a++) { //move all beads
                        Atom positions;
                        positions.x = current.seq[a].pos.x;
                        positions.y = current.seq[a].pos.y;
                        positions.z = current.seq[a].pos.z;

                        //Rotate, tilt, translate --> translations must be the last because rotations are done relative to 0,0,0.

                        positions.rotate(axis_y, deg_rot);
                        positions.rotate(axis_x, deg_tilt);
                        positions.z = positions.z+disp;

                        //calculate deltaG for new positions
                        // double en= current.seq[a].get_E_human(positions.z, E[res_id[current.seq[a].type]]);

                        int inde = res_id[current.name[a]];
                        //cout << i<< " " <<j<< " "<<k<< " " << current.name[a]<<endl;

                        double en_h = aa.seq[inde].get_E_human(positions.z);
                        double en_b = aa.seq[inde].get_E_bacteria(positions.z);

                        total_en_h += en_h;
                        total_en_b += en_b;
                    }
                    double B_en = pow(e, -total_en_h*cte);
                    energies_h[k].push_back(B_en);

                    double B_en_b = pow(e, -total_en_b*cte);
                    energies_b[k].push_back(B_en_b);

                    total_B_en += B_en;
                    total_B_en_b += B_en_b;
                }
            }
        }

        double G, Gb;
        double total_p = 0;
        for (int q= 0; q < winds; q++) {

            for (int p=0; p < energies_h[q].size(); p++) {
                double prob = energies_h[q][p]/total_B_en;
                double probB = energies_b[q][p]/total_B_en_b;

                prob_sumB += probB;

                prob_sum += prob;
            }

            total_p += prob_sum;
            G = -cte2*log(prob_sum);
            G_all.push_back(G);

            Gb = -cte2*log(prob_sumB);
            G_allB.push_back(Gb);

            if (q == (winds-1)) {
                shift = G;
                shiftB = Gb;
            }
            prob_sum = 0;
            prob_sumB = 0;
        }
    double z;
    ofstream myfile2;
    myfile2.open(fileH);
    ofstream myfile3;
    myfile3.open(fileB);

    for( int r = 0; r < winds; r++) {
        z = r*0.1*width;
        myfile2 << z << " " << (G_all[r]-shift) << endl;
        myfile3 << z << " " << (G_allB[r]-shiftB) << endl;
    }

    myfile2.close();
    myfile3.close();
    }

    void print_Emap(Peptide& current, Peptide aa, map<char, int> res_id, string name) {
        Atom axis_y = Atom(0,1,0);
        Atom axis_x = Atom(1,0,0);
        vector<vector<double>> energies_h;
        energies_h.resize(depth, vector<double> (rot_deg*tilt_deg/5));
        vector<vector<double>> energies_b;
        energies_b.resize(depth, vector<double> (rot_deg*tilt_deg/5));
        vector<double> G_all;
        double min=999999;
        double total_B_en = 0;
        double total_B_en_b = 0;
        ofstream myfile, myfile2,myfile3, myfile4;
        string fileH = name + "_H";
        string fileB = name + "_B";
        myfile.open (fileH);
        //myfile2.open ("P_mapH");
        myfile3.open (fileB);
        //myfile4.open ("P_mapB");

        myfile << depth<<" ";
        //myfile2 << depth<<" ";
        myfile3 << depth<<" ";
        //myfile4 << depth<<" ";

        for (int n = 0; n < depth; n++) {
            myfile << n << " ";
            //myfile2 << n << " ";
            myfile3 << n << " ";
            //myfile4 << n << " ";
        }

        myfile<<endl;
        //myfile2<<endl;
        myfile3<<endl;
        //myfile4<<endl;
        double total_en_h = 0;
        double total_en_b = 0;
        for (int i = 0; i < rot_deg ; i=i+15) {
            double deg_rot = i*degToRad;

            for(int j= 0; j< tilt_deg; j = j+15) {
                double deg_tilt = j*degToRad;

                myfile<<j<<i<<" ";
                //myfile2<<j<<i<<" ";
                myfile3<<j<<i<<" ";
                //myfile4<<j<<i<<" ";
                for (int k=0; k < depth; ++k) {
                    double disp = k*0.1;
                    
                    total_en_h = 0;
                    total_en_b = 0;

                    // MOve rotates and get E under peptide class
                    for (int a = 0; a<current.size(); ++a) {
                         Atom positions;
                        positions.x=current.seq[a].pos.x;
                        positions.y=current.seq[a].pos.y;
                        positions.z=current.seq[a].pos.z;

                        //Rotate, tilt, translate -->traslations must be the last because rotations are done relative to 0,0,0.

                        positions.rotate(axis_y, deg_rot);
                        positions.rotate(axis_x, deg_tilt);
                        positions.z=positions.z+disp;

                        //calculate deltaG for new positions
                        //double en=current.seq[a].get_E_human(positions.z, E[res_id[current.seq[a].type]]);

                        int inde=res_id[current.name[a]];
                        //cout<<i<<" "<<j<<" "<<k<<" "<<current.name[a]<<endl;

                        double en_h=aa.seq[inde].get_E_human(positions.z);
                        double en_b=aa.seq[inde].get_E_bacteria(positions.z);

                        total_en_h+=en_h;
                        total_en_b+=en_b;
                    }

                    double B_en=pow(e,-total_en_h*cte);
                    energies_h[k].push_back(B_en);

                    double B_en_b=pow(e,-total_en_b*cte);
                    energies_b[k].push_back(B_en_b);

                    total_B_en+=B_en;
                    total_B_en_b+=B_en_b;
                    myfile<<total_en_h<<" ";
                   // myfile2<<B_en/current.totalP_h<<" ";
                    myfile3<<total_en_b<<" ";
                    //myfile4<<B_en_b/current.totalP_b<<" ";

                }

                myfile<<endl;
                //myfile2<<endl;
                myfile3<<endl;
                //myfile4<<endl;
            }
        }
        myfile.close();
        //myfile2.close();
        myfile3.close();
        //myfile4.close();
    }
};

#endif // ENERGY_CALCULATOR_H


