#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <time.h>
#include <string.h>

// taille du domaine
const int Nx = 64 ;
const int Ny = 64 ;
const int GridSize = Nx*Ny ;

// temps de simulation [s] et de reference
const float T = 2.e+2;

// boite de Lx*Ly [m]
const float Lx = 100.e-9;
const float Ly = 100.e-9;
// longueur de reference [m]
const float L = Lx;
const float L2 = L*L;

// energie de reference [J.m-3]
const float E = 1.e+9;

// diagramme de phase

// concentrations
const float C1_ALPHA = 0.1;
const float C1_BETA = 0.5;

// courbures energie libre J.m-3
const float K1_ALPHA = 1.e+9 / E;
const float K1_BETA = 2.e+9 / E;

const float C2_ALPHA = 0.1;
const float C2_BETA = 0.5;

const float K2_ALPHA = 1.e+9 / E;
const float K2_BETA = 1.e+9 / E;

// coefficients de diffusion [m2.s-1]
const float D1_ALPHA = 1.e-18 * T / L2;
const float D1_BETA = 1.e-17 * T / L2;
const float D2_ALPHA = 1.e-18 * T / L2;
const float D2_BETA = 1.e-17 * T / L2;

// energie d'interface [J.m-2]
const float GAMMA = 1. / (E*L);
// epaisseur d'interface [m]
const float DELTA = 10.e-9 / L;
const float Z = 2.9444389791664403;

// calcul
const int INCREMENTS = 400;
const float DT = T / INCREMENTS / T; 
const float DX = Lx/(float)(Nx-1) / L;
const float DY = Ly/(float)(Ny-1) / L;
const float DX2 = DX*DX;
const float DY2 = DY*DY;

const float FLOAT1 = 1.;
const float FLOAT2 = 2.;
const float FLOAT3 = 3.;
const float FLOAT6 = 6.;
const float FLOAT_HALF = 0.5;

// grandeurs maximales relatives a la carte graphique utilisee ( ici 8800 GT )
const int MaxThread = 512;
const int Thread = MaxThread ;

__global__ void time_increment(float* phi, float* c1, float* c2)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < GridSize)
    {
      int bottom = (tid-Nx)%GridSize;
      int right = tid-tid%Nx+(tid+1)%(Nx);
      int top = (tid+Nx)%GridSize;
      int left = tid-tid%Nx+(tid-1)%(Nx);
      
      if (bottom<0)
	bottom += GridSize;
      if (left<0)
	left+=Nx;
      
      float delta_phi = (phi[right]+phi[left]-FLOAT2*phi[tid])/DX2 + (phi[top]+phi[bottom]-FLOAT2*phi[tid])/DY2;
      float delta_c1 = (c1[right]+c1[left]-FLOAT2*c1[tid])/DX2 + (c1[top]+c1[bottom]-FLOAT2*c1[tid])/DY2;
      float delta_c2 = (c2[right]+c2[left]-FLOAT2*c2[tid])/DX2 + (c2[top]+c2[bottom]-FLOAT2*c2[tid])/DY2;
      
      float c1_beta =  C1_BETA + K1_ALPHA/K1_BETA*(c1[tid] - C1_ALPHA);
      float c2_beta =  C2_BETA + K2_ALPHA/K2_BETA*(c2[tid] - C2_ALPHA);

      float mu1 = K1_ALPHA*(c1[tid] - C1_ALPHA);
      float mu2 = K2_ALPHA*(c2[tid] - C2_ALPHA);
      float F = FLOAT_HALF*( K1_ALPHA * pow( c1[tid] - C1_ALPHA ,2) - K1_BETA * pow( c1_beta - C1_BETA ,2) \
			     + K2_ALPHA * pow( c2[tid] - C2_ALPHA ,2) - K2_BETA * pow( c2_beta - C2_BETA ,2)) \
	- mu1*(c1[tid] - c1_beta) - mu2*(c2[tid] - c2_beta); \

      float DG = FLOAT2*phi[tid]*(1.-phi[tid])*(FLOAT1-FLOAT2*phi[tid]);
      float H = pow(phi[tid],2)*(FLOAT3-FLOAT2*phi[tid]);
      float DH = FLOAT6*phi[tid]*(FLOAT1-phi[tid]);
      float W = FLOAT3*GAMMA*( FLOAT2*Z/DELTA * delta_phi + DELTA/Z * DG );
      
      float D1 = H*D1_ALPHA + (FLOAT1-H)*D1_BETA;
      float D2 = H*D2_ALPHA + (FLOAT1-H)*D2_BETA;

      float MOBILITY = 1.e+15;
      float MOBILITY_N = MOBILITY * ( E * T );

      c1[tid] += DT*D1*delta_c1;
      c2[tid] += DT*D2*delta_c2;
      
      phi[tid] += 0.*1.e+10*DT*(W + DH*F) ;
    }
}

int main(void)
{  
  clock_t start = clock();
  
  cudaFree(0);
  
  // CPU vectors
  float *Host_Phi, *Host_C1, *Host_C2;

  // allocate memory for cpu vectors
  Host_Phi = (float*)malloc( GridSize*sizeof(float) ) ;
  Host_C1 = (float*)malloc( GridSize*sizeof(float) ) ;
  Host_C2 = (float*)malloc( GridSize*sizeof(float) ) ;

  // GPU vectors
  float *dev_Phi, *dev_C1, *dev_C2;

  // allocate memory for gpu vectors
  cudaMalloc( (void **)&dev_Phi, GridSize*sizeof(float) ) ;
  cudaMalloc( (void **)&dev_C1, GridSize*sizeof(float) ) ;
  cudaMalloc( (void **)&dev_C2, GridSize*sizeof(float) ) ;

  // initialize cpu vectors
  for( int i = 0; i < GridSize; i++ )
    {
      float x = (i%Nx) * DX;
      float y = ((i-i%Ny)/Ny) * DY;
      
      float r = sqrt(pow(x-0.5,2) + pow(y-0.5,2)) - 0.25;

      Host_Phi[i] = 0.5 * (1. - tanh(r*Z/DELTA));

      float H = pow(Host_Phi[i],2)*(3.-2.*Host_Phi[i]);

      Host_C1[i] = H*(C1_ALPHA+0.3) + (1.-H)*C1_BETA;
      Host_C2[i] = H*C2_ALPHA + (1.-H)*C2_BETA;

      Host_Phi[i] = 1.;

    }

  // boucle temporelle - debut
  for (int i = 0; i<INCREMENTS; i++)
    {
      printf("incr: %d \n", i) ;
      // Copy host vectors to device
      cudaMemcpy( dev_Phi, Host_Phi, GridSize*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy( dev_C1,  Host_C1,  GridSize*sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy( dev_C2,  Host_C2,  GridSize*sizeof(float), cudaMemcpyHostToDevice);

      // utilisation du kernel
      time_increment <<< 8, Thread  >>> (dev_Phi , dev_C1 , dev_C2) ;

      // copie mem gpu -> cpu
      cudaMemcpy( Host_Phi, dev_Phi, GridSize*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy( Host_C1,  dev_C1,  GridSize*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy( Host_C2,  dev_C2,  GridSize*sizeof(float), cudaMemcpyDeviceToHost);

      char filename[256] = "incr_";
      char increment[3];
      sprintf(increment,"%d",i);
      char extension[5] = ".txt";
      
      strcat(filename, increment);
      strcat(filename, extension);

      FILE *file;
      file = fopen( filename, "w");
      // save to a file
      for( int j = 0; j < GridSize; j++ )
        {
          float x = (j%Nx) * DX;
          float y = ((j-j%Ny)/Ny) * DY;

          fprintf(file, "%f \t %f \t %f \t %f \t %f \n", x, y, Host_Phi[j], Host_C1[j], Host_C2[j]);
        }
      fclose(file);
    }
  // boucle temporelle - fin

  // desaffecte l'espace memoire gpu
  cudaFree (dev_Phi);
  cudaFree (dev_C1);
  cudaFree (dev_C2);

  // desaffecte l'espace memoire cpu
  free(Host_Phi);
  free(Host_C1);
  free(Host_C2);

  clock_t end = clock();
  float seconds = (float)(end - start) / CLOCKS_PER_SEC;

  printf("time elapsed: %f s\n", seconds) ;

  return 0;
}
