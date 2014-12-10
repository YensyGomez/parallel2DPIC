/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <fstream>
#include <curand_kernel.h>
#include <cufft.h>
#include <complex.h>
#include "float.h"

float L,LL;

int N, C,itera;

using namespace std;

// función Maxwelliana de la distribución de las partículas.
__device__ float distribution (float vb, float aleatorio, curandState *states)     //generador de distribución maxwelliana para la velocidad
{

  // Genera un valor random v
   float fmax = 0.5 * (1.0 + exp (-2.0 * vb * vb));
   float vmin = - 5.0 * vb;
   float vmax = + 5.0 * vb;
   float v;
   float f;
   float x;
   int Idx = blockIdx.x*blockDim.x + threadIdx.x;

   while(true){
	   v = vmin + ((vmax - vmin) * aleatorio);
	   f = 0.5 * (exp (-(v - vb) * (v - vb) / 2.0) +
			    exp (-(v + vb) * (v + vb) / 2.0));
	   x = fmax * aleatorio;
	   if(x > f) aleatorio = curand_uniform(states + Idx);
	   else return v;
   }

 return 0;
}
//Distribución aleatoria de las partículas.
__global__ void distribucionParticulas(float *rx,float *ry,float *vx,float *vy,int N,curandState *states,float vb,float L, int seed){
	int Idx = blockIdx.x*blockDim.x + threadIdx.x;

	seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if(Idx < N){
		rx[Idx] = L*curand_uniform(states + Idx);    //inicializando la posicion aleatoria en x
		ry[Idx] = L*curand_uniform(states + Idx);
		vx[Idx] = distribution(vb,curand_uniform(states + Idx),states);//;L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana
		vy[Idx] = distribution(vb,curand_uniform(states + Idx),states);//L*curand_uniform_float(states + Idx);//distribution(vb,states);                          //inicializa la velocidad con una distribucion maxwelliana

	}


}
// inicialización de la densidad.
__global__ void inicializacionDensidad(float *ne,int C){
	int Id=blockIdx.x*blockDim.x + threadIdx.x;
	if(Id<(C*C)){
		ne[Id] = 0.0;
	}
 }

__global__ void inicializacionValoresReales(float *vr,int C){
	int Id=blockIdx.x*blockDim.x + threadIdx.x;
	if(Id<(C*C)){
		vr[Id] = 0.0;
	}
 }

//Calculo de la densidad en cada celda.

__global__ void calculoDensidadInicializacionCeldas(float *rx, float *ry, int *jx,int *jy,float *yx, int N, int C,float L){
	int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 float dx = L / float (C);
	 //float dxx = L /float(C*C);
	if(Id < N){
		 jx[Id] = int(rx[Id]/dx); //posicion en x de la particula
		 jy[Id] = int(ry[Id]/dx); //posicion en y de la particula
		 yx[Id] = (rx[Id]/dx) - (float)jx[Id]; //posicion exacta de la particula en x de la celda "j"
    }

}
__global__ void calculoDensidad(float *ne, int *jx, int *jy,float *yx, int C, float L, int N){
	 float dxx = L /float(C*C);
	// int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 for(int i=0; i<N; i++){
		ne[(jy[i]*C)+jx[i]] += (1. - yx[i])/dxx;
		if(jx[i]+1 == C) ne[(jy[i]*C)] += yx[i]/dxx;
		else ne[(jy[i]*C)+jx[i]+1] += yx[i]/dxx;
	 }

}

__global__ void normalizacionDensidad(float *ne,float *n, int N, int C, float L){
	 int Id = blockIdx.x*blockDim.x + threadIdx.x;
	 if(Id<C*C){
		 n[Id]=float(C*C)*ne[Id]/float(N)-1;
	 }

}

// función que integra la densidad normalizada con la otra densidad
void Output(float *ne_d, float *n_d, int *jx_d,int *jy_d,float *yx_d,int C,float L,int N){
	//definicion de los bloques.
	float blockSize = 1024;
	dim3 dimBlock (ceil(N/blockSize), 1, 1);
	dim3 dimBlock2 (ceil(C*C/blockSize), 1, 1);
	dim3 dimGrid (blockSize, 1, 1);

	calculoDensidad<<<1,1>>>(ne_d,jx_d,jy_d,yx_d,C,L,N);//proceso de mejora.
	cudaDeviceSynchronize();
	normalizacionDensidad<<<blockSize,dimBlock2>>>(ne_d,n_d,N,C,L);
	cudaDeviceSynchronize();


}

//////////////////////////////////////////////////////////////////////////////////////////////////
//Calculo Poisson.

/* en este punto se asignan los valores de la densidad normalizada a una variable compleja que es la que entra a operar con la
 * transformada rápida de fourier en cufft.
 */


__global__ void realTocomplex(float *n_d, cufftComplex *n_d_C, int C){
	int i= blockIdx.x*blockDim.x+threadIdx.x;
	//int j= blockIdx.y*blockDim.y+threadIdx.y;
	//int index= (i*C)+j;// recorrido de la matriz
	if(i<C*C){
		n_d_C[i].x = n_d[i];
		n_d_C[i].y = 0.0f;

	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
/*Normalizacion de la transformada hacia adelante*/

__global__ void normalizacionSalidaTranfForward(cufftComplex *T_F, cufftComplex *T_F_N, int C){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//int j = blockIdx.y*blockDim.y + threadIdx.y;
	//int index= i*C+j;
	  if(i<C*C){
		  T_F_N[i].x=T_F[i].x/float(C*C*C*C);
		  T_F_N[i].y=T_F[i].y/float(C*C*C*C);

	  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void ComplexToFloat2( cufftComplex *T_F_N, float2 *poisson_d,  int C){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<C*C){
	  poisson_d[i].x=T_F_N[i].x;
	  poisson_d[i].y =T_F_N[i].y;

	  //float(1.e+7);

  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
///*Calculo Poisson*/

void Poisson(float2 *calculoPoisson_h, float L, int C){// pasar a calculo paralelo
	float dx = L / float (C);
	calculoPoisson_h[0].x=0.0;
	calculoPoisson_h[0].y=0.0;
	float2 i;
	i.x=0.0;
	i.y=L; //creamos una variable compleja para poder aplicar la discretizacion.
	float2 W;
	W.x= exp(2.0 * M_PI * i.x / float(C));
	W.y= exp(2.0 * M_PI * i.y / float(C));
	float2 Wm;
	Wm.x= L;
	Wm.y= L;
	float2 Wn;
	Wn.x= L;
	Wn.y= L;
	for (int m = 0; m < C; m++)
	{
		for (int n = 0; n < C; n++)
		{
			float2 denom;
			denom.x= 4.0;
			denom.y= 4.0;
			denom.x -= Wm.x + L / Wm.x + Wn.x + L / Wn.x;
			denom.y -= Wm.y + L / Wm.y + Wn.y + L / Wn.y;//se calcula el denominador para cada celda, segun el equema de discretizacion
			if (denom.x!= 0.0 && denom.y!= 0.0){
				calculoPoisson_h[m*C+n].x *= dx *dx / denom.x;
				calculoPoisson_h[m*C+n].y *= dx *dx / denom.y;
			}
			Wn.x *= W.x;//se multiplica por la constante W
			Wn.y *= W.y;
		}
		Wm.x *= W.x;
		Wm.y *= W.y;
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void float2ToComplex(float2 *calculoPoisson_d, cufftComplex *Phi_Poisson, int C){

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<C*C){
	  Phi_Poisson[i].x = calculoPoisson_d[i].x;
	  Phi_Poisson[i].y = calculoPoisson_d[i].y;
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void ComplexToReal(cufftComplex *T_I, float *poissonFinal_d, int C){

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	  if(i<C*C){
		  poissonFinal_d[i] =T_I[i].x/float(1.e-6);

	  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Calculo campo electrico.

__global__ void ElectricBordes(float *poissonFinal_d, float *Ex, float *Ey, float L, int C) // recibe el potencial electroestatico calculado por la funcion poisson  y se calcula el campo electrico, tanto para X como para Y
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  float dx = L / float (C); // el delta de x representa el tamano de la malla

  if(i<C){
	  Ex[i*C]=(poissonFinal_d[((i+1)*C)-1] - poissonFinal_d[(i*C)+1])/(2. * dx);// hallando el campo en x, en la primera columna
	  Ex[((i+1)*C)-1] = (poissonFinal_d[((i+1)*C)-2] - poissonFinal_d[(i*C)]) / (2. * dx);// hallando el campo en x, en la ultima columna
	  Ey[((C-1)*C)+i] = (poissonFinal_d[((C-2)*C)+i] - poissonFinal_d[i]) / (2. * dx); //hallando el campo en "y" para la ultima fila
	  Ey[i] = (poissonFinal_d[((C-1)*C)+i] - poissonFinal_d[i+C]) / (2. * dx);//hallando el campo para la primera fila y la ultima
  }

}

__global__ void calculoCampoElectricoX(float *poissonFinal_d, float *Ex, float L, int C){
	 int i = blockIdx.x*blockDim.x + threadIdx.x;
	 int j = blockIdx.y*blockDim.y + threadIdx.y;
	 float dx = L / float (C); // el delta de x representa el tamano de la malla
	 if(i<C && j<C-2){
		 Ex[j+(C*i)] = (poissonFinal_d[j-1] - poissonFinal_d[j+1]) / (2. * dx);
	 }

}

__global__ void calculoCampoElectricoY(float *poissonFinal_d, float *Ey, float L, int C){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float dx = L / float (C); // el delta de x representa el tamano de la malla
	if(i<((C*C)-C)){
		 Ey[i] = (poissonFinal_d[i-C] - poissonFinal_d[i+C]) / (2. * dx);

	}

}


//////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(){
	// Parametros
	L = 64.0;            // dominio de la solucion 0 <= x <= L (en longitudes de debye)
	//L=LL*LL;
	N = 10000;            // Numero de particulas
	C = 64;            // Número de celdas.
	float vb = 3.0;    // velocidad promedio de los electrones
	//double kappa = 2. * M_PI / (L);
	//float dt=0.1;    // delta tiempo (en frecuencias inversas del plasma)
	//float tmax=10000;  // cantidad de iteraciones. deben ser 100 mil segun el material
	//int skip = int (tmax / dt) / 10; //saltos del algoritmo para reportar datos
	//int itera=0;
	 float salida=0.0;
	 float dx = L / float (C);

/////////////////////////////////////////////////////////////////////////////////////////////////////
//Inicializacion de la posición de las particulas en x, y y velocidad en vx,vy del host y dispositivo.
	float *rx_h,*ry_h,*vx_h,*vy_h;
	float *rx_d,*ry_d, *vx_d,*vy_d;
	int *jx_d, *jy_d;
	float *yx_d;
////////////////////////////////////////////////////////////////////////////////////////////////////
	// inicialización de las variables de densidad del host y dispositivo.
	float *ne_h;
	float *ne_d;
	float *n_h; // densidad normalizada.
	float *n_d; // densidad normalizada del dispositivo.
	float2 *calculoPoisson_h;
	float2 *calculoPoisson_d;
	float  * poissonFinal_h;
	float  * poissonFinal_d;
	float *Ex_h;
	float *Ey_h; //campoElectrico
	float *Ex_d;
	float *Ey_d; // Campo Electrico en el dispositivo.



////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Crear la variable tipo cufftComplex*/
	cufftComplex *n_d_C; // covertir la densidad en una variable compleja.
	cufftComplex *T_F;	 // primer paso de la transformada hacia adelante
	cufftComplex *T_F_N; //Trasformada hacia adelante normalizada.
	cufftComplex *Phi_Poisson; // esta variable muestra la solucion de poisson.
	cufftComplex *T_I;   // Transformada Inversa.

////////////////////////////////////////////////////////////////////////////////////////////////////
	int size = N*sizeof(float);
	int size_ne = C*C*sizeof(float);
	int size_ne2 = C*C*sizeof(float2);

//////////////////////////////////////////////////////////////////////////////////////////////////////
	//reserva en memoria al host
	rx_h = (float *)malloc(size);
	ry_h = (float *)malloc(size);
	vx_h = (float *)malloc(size);
	vy_h = (float *)malloc(size);
	ne_h = (float *)malloc(size_ne);
	n_h = (float *)malloc(size_ne);
	calculoPoisson_h=(float2 *)malloc(size_ne2);
	poissonFinal_h=(float *)malloc(size_ne);
	Ex_h = (float *)malloc(size_ne);
	Ey_h = (float *)malloc(size_ne);



//////////////////////////////////////////////////////////////////////////////////////////////////////
	//reserva de memoria del dispositivo.
	cudaMalloc((void **)&rx_d,size);
	cudaMalloc((void **)&ry_d,size);
	cudaMalloc((void **)&vx_d,size);
	cudaMalloc((void **)&vy_d,size);
	cudaMalloc((void **)&ne_d,size_ne);
	cudaMalloc((void **)&n_d,size_ne);
	cudaMalloc((void **)&jx_d,size);
	cudaMalloc((void **)&jy_d,size);
	cudaMalloc((void **)&yx_d,size);
	cudaMalloc((void **)&calculoPoisson_d,size_ne2);
	cudaMalloc((void **)&poissonFinal_d,size_ne);
	cudaMalloc((void **)&Ex_d,size_ne);
	cudaMalloc((void **)&Ey_d,size_ne);


/////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*Asignación de memoria a la variable tipo cufftComplex */
	cudaMalloc((void **)&n_d_C,sizeof(cufftComplex)*C*C);
	cudaMalloc((void **)&T_F,sizeof(cufftComplex)*C*C);
	cudaMalloc((void **)&T_F_N,sizeof(cufftComplex)*C*C);
	cudaMalloc((void **)&Phi_Poisson,sizeof(cufftComplex)*C*C);
	cudaMalloc((void **)&T_I,sizeof(cufftComplex)*C*C);
//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//valores aleatorios y tamaños de los vectores.
	curandState *devStates;
	cudaMalloc((void **) &devStates, N * sizeof(curandState));


	float blockSize = 1024;
	dim3 dimBlock (ceil(N/blockSize), 1, 1);
	dim3 dimBlock2 (ceil(C*C/blockSize), 1, 1);
	dim3 dimBlock3 (ceil(C*C/blockSize), ceil(C*C/blockSize), 1);
	dim3 dimGrid (blockSize, 1, 1);
	dim3 dimGrid3 (blockSize, blockSize, 1);
	int seed = time(NULL);


	distribucionParticulas<<<blockSize,dimBlock>>>(rx_d,ry_d,vx_d,vy_d,N,devStates,vb,L, seed);
	cudaDeviceSynchronize();

	inicializacionDensidad<<<blockSize,dimBlock2>>>(ne_d,C);
	cudaDeviceSynchronize();

	calculoDensidadInicializacionCeldas<<<blockSize,dimBlock>>>(rx_d,ry_d,jx_d,jy_d,yx_d,N,C,L);
	cudaDeviceSynchronize();

	//funcion Calculo densidad.
	Output(ne_d,n_d,jx_d,jy_d,yx_d, C,L,N);     // Calculo de la densidad y normalización de la densidad.
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	realTocomplex<<<blockSize,dimBlock2>>>(n_d, n_d_C,C);
	cudaDeviceSynchronize();

	cufftHandle plan;
	cufftPlan2d(&plan,C,C,CUFFT_C2C);
	cufftExecC2C(plan,n_d_C,T_F,CUFFT_FORWARD); // transformada hacia adelante en x and y.

	/*Valor de la transformada hacia adelante de latransformada rápida normalizada*/

	normalizacionSalidaTranfForward<<<blockSize,dimBlock2>>>(T_F,T_F_N, C);
	cudaDeviceSynchronize();

	/*Calculo Poisson*/

	 ComplexToFloat2<<<blockSize,dimBlock2>>>(T_F_N,calculoPoisson_d,C);
	 cudaDeviceSynchronize();

	 //Comprobacion del resultado de la transformada hacia adelante.
	 cudaMemcpy(calculoPoisson_h, calculoPoisson_d, size_ne2, cudaMemcpyDeviceToHost);

	 //Calculo Poisson antes de la transformada inversa
	 Poisson(calculoPoisson_h,L,C);

	 // Pasar el calculo de Poisson al dispositivo

	 cudaMemcpy(calculoPoisson_d, calculoPoisson_h, size_ne2, cudaMemcpyHostToDevice);

	 //Hacer la transformada Inversa
	 float2ToComplex<<<blockSize,dimBlock2>>>(calculoPoisson_d,Phi_Poisson,C);
	 cudaDeviceSynchronize();

	 //Aplicar la transformada inversa de la matriz

	 cufftExecC2C(plan,Phi_Poisson,T_I,CUFFT_INVERSE);

	 //tomando la transformada final.

	 ComplexToReal<<<blockSize,dimBlock2>>>(T_I,poissonFinal_d,C);
	 cudaDeviceSynchronize();
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	 ElectricBordes<<<blockSize,dimBlock2>>>(poissonFinal_d,Ex_d,Ey_d, L,C); // Campo Electrico en los bordes.
	 cudaDeviceSynchronize();

	 /*Calculo del campo electrico para x*/
	 calculoCampoElectricoX<<<dimGrid3,dimBlock3>>>(poissonFinal_d,Ex_d,L,C); // se utilizan dos hilos de debe organizar la manera como se envian.

	 /*Calculo del campo electrico para y*/
	 calculoCampoElectricoY<<<blockSize,dimBlock2>>>(poissonFinal_d, Ey_d,L,C);
	 cudaDeviceSynchronize();




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//posicion en x.
	cudaMemcpy(rx_h, rx_d, size, cudaMemcpyDeviceToHost);

	// posicion en y.
	cudaMemcpy(ry_h, ry_d, size, cudaMemcpyDeviceToHost);

	// velocidad en x.
	cudaMemcpy(vx_h, vx_d, size, cudaMemcpyDeviceToHost);

	//velocidad en y.
	cudaMemcpy(vy_h, vy_d, size, cudaMemcpyDeviceToHost);
	//inicializacion densidades
	cudaMemcpy(ne_h, ne_d, size_ne, cudaMemcpyDeviceToHost);
	//normalización de la densidad.
	cudaMemcpy(n_h, n_d, size_ne, cudaMemcpyDeviceToHost);
	//calculo de la transformada rapida de fourier despues de la inversa.
	cudaMemcpy(poissonFinal_h, poissonFinal_d, size_ne, cudaMemcpyDeviceToHost);
	//Calculo de Campo Electrico Ex, Ey.
	cudaMemcpy(Ex_h, Ex_d, size_ne, cudaMemcpyDeviceToHost);
	cudaMemcpy(Ey_h, Ey_d, size_ne, cudaMemcpyDeviceToHost);

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	ofstream init;
		init.open("distribucionInicial.txt");
		  		    for (int i = 0; i < N; i++){
		  		    	init<<rx_h[i]<<" "<<ry_h[i]<<" "<<vx_h[i]<<" "<<vy_h[i]<<endl;

		  		    }

		  		    init.close();


		init.open("salida_densidad3.txt");
					for (int i = 0; i < C*C; i++){
						init<<ne_h[i]<<" "<<endl;
						salida+=ne_h[i];
					}

					init.close();
					cout<<salida<<" "<<dx<<endl;

		init.open("densidadNormalizada.txt");
					for (int i = 0; i < C; i++){
						for (int j = 0; j < C; j++){
						init<<n_h[(i*C)+j]<<" ";
						}
					init<<endl;
					}

					init.close();
					cout<<salida<<" "<<dx<<endl;

	init.open("CalculoPoissonAntesdeLaTranformadaInversa.txt");
				for (int i = 0; i < C; i++){
					for (int j = 0; j < C; j++){
					init<<calculoPoisson_h[(i*C)+j].x<<" ";
					}
				init<<endl;
				}

				init.close();

	init.open("DespuesTransformadaInversaPoissonFinal");
					for (int i = 0; i < C; i++){
						for (int j = 0; j < C; j++){
						init<<poissonFinal_h[(i*C)+j]<<" ";
						}
					init<<endl;
					}

					init.close();


	init.open("CamposElectricos");

					for (int i = 0; i < C*C; i++){
						init<<Ex_h[i]<<" "<<Ey_h[i]<<endl;
					}

					init.close();

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				/*Liberar memoria*/
	free(rx_h);
	free(ry_h);
	free(vx_h);
	free(vy_h);
	free(ne_h);
	free(n_h);
	free(calculoPoisson_h);
	free(poissonFinal_h);
	cufftDestroy(plan);
	free(Ex_h);
	free(Ey_h);
	cudaFree(rx_d);
	cudaFree(ry_d);
	cudaFree(vx_d);
	cudaFree(vy_d);
	cudaFree(ne_d);
	cudaFree(n_d);
	cudaFree(n_d_C);
	cudaFree(T_F);
	cudaFree(T_F_N);
	cudaFree(Phi_Poisson);
	cudaFree(T_I);
	cudaFree(calculoPoisson_d);
	cudaFree(poissonFinal_d);
	cudaFree(Ex_d);
	cudaFree(Ey_d);
	return (0);

}
