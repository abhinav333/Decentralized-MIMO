/*****************************************************************************
 *     Author: Xilinx, Inc.

 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.

 *     XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS"
 *     AS A COURTESY TO YOU, SOLELY FOR USE IN DEVELOPING PROGRAMS AND
 *     SOLUTIONS FOR XILINX DEVICES.  BY PROVIDING THIS DESIGN, CODE,
 *     OR INFORMATION AS ONE POSSIBLE IMPLEMENTATION OF THIS FEATURE,
 *     APPLICATION OR STANDARD, XILINX IS MAKING NO REPRESENTATION
 *     THAT THIS IMPLEMENTATION IS FREE FROM ANY CLAIMS OF INFRINGEMENT,
 *     AND YOU ARE RESPONSIBLE FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE
 *     FOR YOUR IMPLEMENTATION.  XILINX EXPRESSLY DISCLAIMS ANY
 *     WARRANTY WHATSOEVER WITH RESPECT TO THE ADEQUACY OF THE
 *     IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OR
 *     REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE FROM CLAIMS OF
 *     INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *     FOR A PARTICULAR PURPOSE.

 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.

 *     (c) Copyright 2013-2014 Xilinx Inc.
 *     All rights reserved.

 *****************************************************************************/

#include "matrix_multiply_fixed.h"

// The top-level functions for each implementation "target": small, balanced, fast, faster, tradeoff
// o Function return must be "void" to use the set_directive_top
/*******************************************/
//ADMM definitions





	/*
	cluster_mat1=np.zeros((Max_C*K,MperC),dtype=H.dtype)
	cluster_mat2=np.zeros((Max_C*K,K),dtype=H.dtype)
	cluster_mat3=np.zeros((Max_C*K,K),dtype=H.dtype)
	*/

//	MATRIX_T temp1,temp2;





//Array estimates
//H matrix: 32 x 4 = 128 complex entries (input)
//y matrix: 32 x 1 =32 complex entries (input)
//x matrix: 4 x 1 = 4 complex entries (output)
/*
w=np.zeros((K,1),dtype=H.dtype)
c=np.zeros((K,Max_C),dtype=H.dtype)
vc=np.zeros((K,Max_C),dtype=H.dtype)
Lambdac=np.zeros((K,Max_C),dtype=H.dtype)
rc=np.zeros((K,1),dtype=H.dtype)
Max_C=cluster
*/

//ADMM-GS algorithm
void matrix_multiply_default(const MATRIX_T A [A_ROWS][A_COLS],
                             const MATRIX_T B [B_ROWS][B_COLS],
                                   MATRIX_T C [C_ROWS][C_COLS],int& inverse_OK){

}
//template<int R1,int R2,int R3>
void cmplx_division(MATRIX_T a[K][1], MATRIX_T b[K][1], MATRIX_T c[K][1])
{
#pragma HLS INLINE
//#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T temp1[K][1];
//#pragma HLS RESOURCE variable=temp1 core=DSP48E
	CMPLX_TYPE temp2[K][1];
#pragma HLS ARRAY_PARTITION variable=temp1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=a complete dim=1
#pragma HLS ARRAY_PARTITION variable=b complete dim=1
#pragma HLS ARRAY_PARTITION variable=c complete dim=1


#pragma HLS pipeline
 col_loooop:for(int r=0;r<K;r++)
	{
#pragma HLS unroll
//#pragma HLS pipeline

#pragma HLS dependence variable=temp2 RAW intra true
#pragma HLS dependence variable=temp1 RAW intra true
#pragma HLS dependence variable=temp2 inter false
#pragma HLS dependence variable=temp1 inter false

#pragma HLS dependence variable=b inter false
#pragma HLS dependence variable=a inter false
#pragma HLS dependence variable=c inter false


	  temp2[r][0]=b[r][0].real();
	 // temp1[r][0].real(a[r][0].real()/temp2[r][0]);
	  temp1[r][0]=a[r][0]/temp2[r][0];
	  c[r][0]=temp1[r][0];
	}
 }

void cmplx_division_r(MATRIX_T a, MATRIX_T b, MATRIX_T c)
{
#pragma HLS INLINE
	MATRIX_T temp1;
	CMPLX_TYPE temp2;

//#pragma HLS dependence variable=temp2 RAW intra true
//#pragma HLS dependence variable=temp1 RAW intra true
//#pragma HLS dependence variable=temp2 inter false
//#pragma HLS dependence variable=temp1 inter false

//#pragma HLS dependence variable=b inter false
//#pragma HLS dependence variable=a inter false
//#pragma HLS dependence variable=c inter false


	  temp2=b.real();
	  temp1=a/temp2;
	  c=temp1;
}


template<int R1, int C1,int R2,int C2>
void custom_matrix_multiply(MATRIX_T am[R1][C1], MATRIX_T bm[R2][C2],MATRIX_T cm[R1][C2]){
#pragma HLS INLINE
MATRIX_T temp1[1][1];
#pragma HLS ARRAY_PARTITION variable=am complete dim=1
#pragma HLS ARRAY_PARTITION variable=bm complete dim=2
#pragma HLS ARRAY_PARTITION variable=cm complete
#pragma HLS pipeline
	a_l1:for(int r=0;r<R1;r++){
		#pragma HLS unroll
		b_l1:for(int c=0;c<C2;c++){
				#pragma HLS unroll
			c_l1:for(int k=0;k<C1;k++){
#pragma HLS DEPENDENCE variable=temp1 intra RAW true
				temp1[0][0]=am[r][k]*bm[k][c];
				cm[r][c]=cm[r][c]+temp1[0][0];
			}
		}
	}

}

//cluster_mat1=Hc;
//cluster_mat2=Hc^{H}Hc;

//Stochastic Quasi-Newton
void f_gradient1(MATRIX_T cluster_mat2[K][K],
				MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K],
				MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
				MATRIX_T x_e[K][1],
				MATRIX_T f1[K][1])
{
#pragma HLS pipeline
#pragma HLS INLINE
#pragma HLS INLINE RECURSIVE

	MATRIX_T temp1[K][1];
	MATRIX_T temp2[K][1];


	hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,K,K,K,1, K, 1, MULT_CONFIG_FASTER_f_g1_1, MATRIX_T, MATRIX_T> (cluster_mat2, x_e, temp1);
	hls::matrix_multiply_top<hls::ConjugateTranspose, hls::NoTranspose,ANTENNA_PER_CLUSTER,K,ANTENNA_PER_CLUSTER, 1, K,1, MULT_CONFIG_FASTER_f_g1_2, MATRIX_T, MATRIX_T> (cluster_mat1, yc, temp2);
	c_row_loop_f1 : for (int r=0;r<C_ROWS_F;r++){
		#pragma HLS pipeline
		      f1[r][0] = temp1[r][0]-temp2[r][0];
	}

}


void f_gradient1_impr(MATRIX_T cluster_mat2[K][K],
				MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
				MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
				MATRIX_T x_e[K][1],
				MATRIX_T f1[K][1])
{
//#pragma HLS pipeline
#pragma HLS INLINE
#pragma HLS INLINE RECURSIVE

	MATRIX_T temp1[K][1][1];
	MATRIX_T temp2[K][1][1];
	MATRIX_T temp3[1][K];
	MATRIX_T temp4[1][ANTENNA_PER_CLUSTER];

#pragma HLS ARRAY_PARTITION variable=temp1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=temp2 complete dim=1

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1

	k_loop:for(int r=0;r<K;r++)
	{
#pragma HLS unroll
//#pragma HLS pipeline
		k1_l: for(int r1=0;r1<K;r1++)
			temp3[0][r1]=cluster_mat2[r][r1];

		k1_2: for(int r1=0;r1<ANTENNA_PER_CLUSTER;r1++)
			temp4[0][r1]=cluster_mat1[r][r1];

		hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,1,K,K,1, 1, 1, MULT_CONFIG_FASTER_f_g1_3, MATRIX_T, MATRIX_T> (cluster_mat2, x_e, temp1[r]);
		hls::matrix_multiply_top<hls::NoTranspose, hls::NoTranspose,1,ANTENNA_PER_CLUSTER,ANTENNA_PER_CLUSTER, 1, 1,1, MULT_CONFIG_FASTER_f_g1_4, MATRIX_T, MATRIX_T> (cluster_mat1, yc, temp2[r]);

		f1[r][0]=temp1[r][0][0]-temp2[r][0][0];



	}


/*
	c_row_loop_f1 : for (int r=0;r<C_ROWS_F;r++){
		#pragma HLS pipeline
		      f1[r][0] = temp1[r][0]-temp2[r][0];
	}
*/
}

void f_gradient2(MATRIX_T cluster_mat2[K][K],
					MATRIX_T f2[K][1])
{
#pragma HLS INLINE
	#pragma HLS INLINE RECURSIVE
	b_row_loop_f2 :  for (int r=0;r<B_ROWS_F;r++)
        #pragma HLS pipeline
		        f2[r][0] = cluster_mat2[r][r];

}

void sqn_ring_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T g2[K][1],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter,int f)
{

#pragma HLS INTERFACE ap_none port=cluster_mat1
#pragma HLS INTERFACE ap_none port=cluster_mat2
#pragma HLS INTERFACE ap_none port=g2
#pragma HLS INTERFACE ap_none port=yc
#pragma HLS INTERFACE ap_none port=clust_number
#pragma HLS INTERFACE ap_none port=iter
#pragma HLS INTERFACE ap_none port=data_path1
#pragma HLS INTERFACE ap_none port=data_path2



//#pragma HLS INLINE
#pragma HLS PIPELINE
#pragma HLS function_instantiate variable=clust_number
#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

	if(iter==0){
			f_gradient2(cluster_mat2,f_g2);
			b_row_loop_f_g2 :  for (int r=0;r<B_ROWS_F;r++){
				#pragma HLS unroll
				data_path1[r][0] = data_path1[r][0]+ f_g2[r][0];
				if(clust_number==(CLUSTERS-1)){
					g2[r][0]=data_path1[r][0];
				}
			}
		 }
/*#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete*/

		f_gradient1_impr(cluster_mat2,cluster_mat1,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS unroll
			data_path2[r][0] = data_path2[r][0] + f_g1[r][0];
			if(clust_number==(CLUSTERS-1)){
				g1_g2_r[r][0].real(data_path2[r][0].real()/g2[r][0].real());
				g1_g2_r[r][0].imag(data_path2[r][0].imag()/g2[r][0].real());
				data_path1[r][0]=data_path1[r][0]-g1_g2_r[r][0];
				data_path2[r][0]=0;
 			}
		}

}

void sqn_ring_cluster_flatten(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					MATRIX_T g2[K][1],
					int clust_number)

{


//#pragma HLS INLINE
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline


	//MATRIX_T g2[K][1];
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];

    	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    	clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];

    	}


    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS pipeline
    				    		    		data_path1_out_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_out_l[r][0] = 0;
    				    		}

    			}
    			else{
    			f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,data_path1_in_l,f_g1);
				b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
									#pragma HLS unroll
				data_path2_out_l[r][0] = data_path2_in_l[r][0] + f_g1[r][0];
				data_path1_out_l[r][0]= data_path1_in_l[r][0];

				}
    		}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}


void sqn_ring_cluster_flatten_system_gen(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					int clust_number)

{
/*
#pragma HLS INTERFACE axis  port=cluster_mat1
#pragma HLS INTERFACE axis  port=cluster_mat2
#pragma HLS INTERFACE axis  port=yc
#pragma HLS INTERFACE axis  port=data_path1_in
#pragma HLS INTERFACE axis  port=data_path2_in
#pragma HLS INTERFACE axis  port=data_path1_out
#pragma HLS INTERFACE axis  port=data_path2_out
*/

#pragma HLS INTERFACE ap_fifo depth=1024 port=cluster_mat1
#pragma HLS INTERFACE ap_fifo depth=1024 port=cluster_mat2
#pragma HLS INTERFACE ap_fifo depth=1024 port=yc
#pragma HLS INTERFACE ap_fifo depth=1024 port=data_path1_in
#pragma HLS INTERFACE ap_fifo depth=1024 port=data_path2_in
#pragma HLS INTERFACE ap_fifo depth=1024 port=data_path1_out
#pragma HLS INTERFACE ap_fifo depth=1024 port=data_path2_out






//#pragma HLS INLINE
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline


	MATRIX_T g2[K][1];
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];

    	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    	clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];

    	}


    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				//cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS unroll
    				    		    		data_path1_in_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_in_l[r][0] = 0;
    				    		}

    			}
    			else{
    			f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,data_path1_in_l,f_g1);
				b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
									#pragma HLS unroll
				data_path2_out_l[r][0] = data_path2_in_l[r][0] + f_g1[r][0];
				data_path1_out_l[r][0]= data_path1_in_l[r][0];

				}
    		}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}
/*
void sqn_ring_cluster_flatten_frames(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
						MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[FRAMES][ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1_in[FRAMES][K][1],
					MATRIX_T data_path2_in[FRAMES][K][1],
					MATRIX_T data_path1_out[FRAMES][K][1],
					MATRIX_T data_path2_out[FRAMES][K][1],
					int clust_number)

{


//#pragma HLS INLINE
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline


	MATRIX_T g2[K][1];
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];

    	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    	clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];

    	}


    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS unroll
    				    		    		data_path1_in_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_in_l[r][0] = 0;
    				    		}

    			}
    			else{
    			f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,data_path1_in_l,f_g1);
				b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
									#pragma HLS unroll
				data_path2_out_l[r][0] = data_path2_in_l[r][0] + f_g1[r][0];
				data_path1_out_l[r][0]= data_path1_in_l[r][0];

				}
    		}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}
*/
void sqn_ring_cluster_flatten_acc(MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					MATRIX_T g2[K][1],
					int clust_number)

{
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS pipeline

	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	CMPLX_TYPE temp1[K][1];
	CMPLX_TYPE temp2[K][1];

	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	static MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
   /* get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];
    	}*/

    		cmplx_division(data_path2_in,g2,g1_g2_r);
    		row_loop_new:  for (int r=0;r<K;r++){
    		    		#pragma HLS unroll
    		    		data_path1_out[r][0] = data_path1_in[r][0]-g1_g2_r[r][0];
    		    		data_path2_out[r][0] = 0;
    		}


  			/*set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];
									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}*/

}




void sqn_ring_frame(MATRIX_T y[M][1], MATRIX_T x[K][1]){


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[CLUSTERS+1][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS+1][K][K];
MATRIX_T yc[CLUSTERS+1][ANTENNA_PER_CLUSTER][1];


		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[cls][r][0]=y[cls*CLUSTERS+r][0];



MATRIX_T g2_f1[K][1];
static MATRIX_T data_path1_in[CLUS1+2][K][1];
static MATRIX_T data_path2_in[CLUS1+2][K][1];
static MATRIX_T data_path1_out[CLUS1+2][K][1];
static MATRIX_T data_path2_out[CLUS1+2][K][1];
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1



#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
//#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
//#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
//#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
//#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P


//#pragma HLS pipeline
it_1: for(int it=0;it<CLUS1;it++)
	{

	#pragma HLS unroll
#pragma HLS DEPENDENCE variable=data_path2_out inter RAW true
#pragma HLS DEPENDENCE variable=data_path1_out inter RAW true
#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true
#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

 // if(it%(CLUSTERS+1)==CLUSTERS)
  //{
//	  sqn_ring_cluster_flatten_acc(data_path1_in[it],data_path2_in[it],data_path1_out[it],data_path2_out[it],g2_f1,it);
  //}
  //else{
	  //sqn_ring_cluster_flatten(cluster_mat1[it%(CLUSTERS+1)],cluster_mat2[it%(CLUSTERS+1)],yc[it%(CLUSTERS+1)],data_path1_in[it],data_path2_in[it],data_path1_out[it],data_path2_out[it],it);
//  }
	r_l_data:for(int r=0;r<K;r++){
							#pragma HLS pipeline
								data_path1_in[it+1][r][0]=data_path1_out[it][r][0];
								data_path2_in[it+1][r][0]=data_path2_out[it][r][0];
								}


		}

		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS pipeline
			x[r][0]=data_path1_out[CLUS-1][r][0];



}

void sqn_ring_frame_throughput(MATRIX_T y[FRAMES][M][1], MATRIX_T x[FRAMES][K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[FRAMES][CLUSTERS+1][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[FRAMES][CLUSTERS+1][K][K];
MATRIX_T yc[FRAMES][CLUSTERS+1][ANTENNA_PER_CLUSTER][1];




MATRIX_T g2_f1[FRAMES][K][1];
static MATRIX_T data_path1_in[FRAMES][CLUS1+2][K][1];
static MATRIX_T data_path2_in[FRAMES][CLUS1+2][K][1];
static MATRIX_T data_path1_out[CLUS1+2][K][1];
static MATRIX_T data_path2_out[CLUS1+2][K][1];
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1

#pragma HLS ARRAY_PARTITION variable=yc complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2


#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P





	frames_loop_1:for(int f=0;f<FRAMES;f++)
				#pragma HLS unroll
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[f][cls][r][0]=y[f][cls*ANTENNA_PER_CLUSTER+r][0]; //Need to change this cls*ANTENNA_PER_CLUSTER

frames_loop_2:for(int f=0;f<FRAMES;f++){
#pragma HLS unroll
#pragma HLS DEPENDENCE variable=yc inter false



it_1: for(int it=0;it<CLUS1;it++)
{
#pragma HLS loop_flatten off
#pragma HLS unroll
#pragma HLS DEPENDENCE variable=yc inter false

#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
#pragma HLS DEPENDENCE variable=g1_f1 inter false
#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true
	//if((it%(CLUSTERS+1))==CLUSTERS){

		//sqn_ring_cluster_flatten_acc(data_path1_in[f][it],data_path2_in[f][it],data_path1_in[f][it+1],data_path2_in[f][it+1],g2_f1[f],it);
	//}
	//else
	//	{
		sqn_ring_cluster_flatten(cluster_mat1[f][it%(CLUSTERS+1)],cluster_mat2[f][it%(CLUSTERS+1)],yc[f][it%(CLUSTERS+1)],data_path1_in[f][it],data_path2_in[f][it],data_path1_in[f][it+1],data_path2_in[f][it+1],g2_f1[f],it);
	//	}
	/*r_l_data:for(int r=0;r<K;r++){
								#pragma HLS pipeline
									data_path1_in[it+1][r][0]=data_path1_out[it][r][0];
									data_path2_in[it+1][r][0]=data_path2_out[it][r][0];
									}*/

 }
}
	frames_loop_3:for(int f=0;f<FRAMES;f++)
			#pragma HLS unroll
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[f][r][0]=data_path1_in[f][CLUS][r][0];
}




void sqn_ring_throughput(MATRIX_T y[M][1], MATRIX_T x[K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[CLUSTERS+1][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS+1][K][K];
MATRIX_T yc[CLUSTERS+1][ANTENNA_PER_CLUSTER][1];




MATRIX_T g2_f1[K][1];
static MATRIX_T data_path1_in[CLUS1+2][K][1];
static MATRIX_T data_path2_in[CLUS1+2][K][1];
static MATRIX_T data_path1_out[CLUS1+2][K][1];
static MATRIX_T data_path2_out[CLUS1+2][K][1];
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1



#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P





		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[cls][r][0]=y[cls*CLUSTERS+r][0];





it_1: for(int it=0;it<CLUS1;it++)
{

#pragma HLS unroll
#pragma HLS DEPENDENCE variable=yc inter false

#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

	//sqn_ring_cluster_flatten(cluster_mat1[it%(CLUSTERS+1)],cluster_mat2[it%(CLUSTERS+1)],yc[it%(CLUSTERS+1)],data_path1_in[it],data_path2_in[it],data_path1_in[it+1],data_path2_in[it+1],it);

}

		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[r][0]=data_path1_in[CLUS][r][0];
}

void sqn_ring_frame_experi(MATRIX_T y[M][1], MATRIX_T x[K][1])
{


#pragma HLS INTERFACE ap_fifo port=y
#pragma HLS INTERFACE ap_fifo port=x

MATRIX_T cluster_mat1[CLUSTERS+1][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS+1][K][K];
MATRIX_T yc[CLUSTERS+1][ANTENNA_PER_CLUSTER][1];




MATRIX_T g2_f1[K][1];
static MATRIX_T data_path1_in[CLUS1+2][K][1];
static MATRIX_T data_path2_in[CLUS1+2][K][1];
static MATRIX_T data_path1_out[CLUS1+2][K][1];
static MATRIX_T data_path2_out[CLUS1+2][K][1];
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_out complete dim=1
#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1



#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P









		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[cls][r][0]=y[cls*CLUSTERS+r][0];

#pragma HLS pipeline
it_1: for(int it=0;it<CLUS1;it++)
{

#pragma HLS unroll
//#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
//#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true
#pragma HLS DEPENDENCE variable=yc inter false
#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
#pragma HLS DEPENDENCE variable=cluster_mat2 inter false

	//sqn_ring_cluster_flatten(cluster_mat1[it%(CLUSTERS+1)],cluster_mat2[it%(CLUSTERS+1)],yc[it%(CLUSTERS+1)],data_path1_in[it],data_path2_in[it],data_path1_in[it+1],data_path2_in[it+1],it);

	}



		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[r][0]=data_path1_in[CLUS-1][r][0];



}
///////////////////////////////////////////////STAR ALgorithm\\\\\\\\\\\\\\\\\\

void sqn_star_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter)
{
#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];

	get_local_data_path:{
	#pragma HLS loop_merge
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d1_temp_in[r][0] = data_path1[r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d2_temp_in[r][0] = data_path2[r][0];

   	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];


	}

//	if(iter==0){
	//	f_gradient2(cluster_mat2,d1_temp_out);
	//}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			d2_temp_out[r][0] = f_g1[r][0];
		}


		set_local_data_path:{
			#pragma HLS loop_merge
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path1[r][0]=d1_temp_out[r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path2[r][0]=d2_temp_out[r][0];
			}
}


void sqn_star_cluster_flatten(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					int clust_number)

{
//#pragma HLS INLINE
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline


	MATRIX_T g2[K][1];
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];

    	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    	clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];

    	}


    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS unroll
    				    		    		data_path1_in_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_in_l[r][0] = 0;
    				    		}

    			}
    			else{
    			f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,data_path1_in_l,f_g1);
				b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
									#pragma HLS unroll
				data_path2_out_l[r][0] = data_path2_in_l[r][0] + f_g1[r][0];
				data_path1_out_l[r][0]= data_path1_in_l[r][0];

				}
    		}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}

#if 0
void sqn_star_apex_cluster(MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K],
		MATRIX_T cluster_mat2[K][K],
		MATRIX_T g2[K][1],
		MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
		MATRIX_T data_path1[CLUSTERS-1][K][1],
		MATRIX_T data_path2[CLUSTERS-1][K][1],
		int iter)
{
#pragma HLS INLINE RECURSIVE

		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
/*
		if(iter==0){
				f_gradient2(cluster_mat2,f_g2);
				b_row_loop_f_g2 :  for (int r=0;r<B_ROWS_F;r++){
					c_column_f_g2: for(int c=0;c<CLUSTERS-1;c++){
					#pragma HLS pipeline
						g2[r][0]=g2[r][0]+data_path1[c][r][0];
					}
					g2[r][0]=g2[r][0]+f_g2[r][0];
				}
	     }*/
/*
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete*/
		f_gradient1_impr(cluster_mat2,cluster_mat1,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			c_column_f_g1: for(int c=0;c<CLUSTERS-1;c++){
				#pragma HLS pipeline
				f_g1[r][0]=f_g1[r][0]+data_path2[c][r][0];
			}
			g1_g2_r[r][0].real(f_g1[r][0].real()/g2[r][0].real());
			g1_g2_r[r][0].imag(f_g1[r][0].imag()/g2[r][0].real());
			d1_temp_out[r][0]=d1_temp_in[r][0]-g1_g2_r[r][0];



		}

}
#endif

void sqn_star_apex_cluster1(MATRIX_T g2[K][1],
		MATRIX_T data_path1[CLUSTERS][K][1],
		MATRIX_T data_path2[CLUSTERS][K][1],
		MATRIX_T x_init[K][1],
		int iter)

{
#pragma HLS function_instantiate variable=iter
#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline
		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
	    MATRIX_T g2_temp[K][1];


		#pragma HLS ARRAY_PARTITION variable=g1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g1_g2_r complete dim=1
		#pragma HLS ARRAY_PARTITION variable=x_init complete dim=1
		#pragma HLS ARRAY_PARTITION variable=d1_temp_out complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2

/*		if(iter==0){
			 b_row_loop_f_g2_1 : for(int r=0;r<K;r++){
				#pragma HLS pipeline
					c_column_f_g2_1: for(int c=0;c<CLUSTERS;c++){
						g2_temp[r][0]=g2_temp[r][0]+data_path1[c][r][0];
					}
				 d1_temp_in[r][0]=x_init[r][0];
				}
	     }*/

			b_row_loop_f_g1 :  for (int r=0;r<K;r++){
	/*		#pragma HLS DEPENDENCE variable=g1 intra RAW true
			#pragma HLS DEPENDENCE variable=g1_g2_r intra RAW true
			#pragma HLS DEPENDENCE variable=d1_temp_out intra RAW true
			#pragma HLS DEPENDENCE variable=x_init intra WAR true
			#pragma HLS DEPENDENCE variable=g1_g2_r inter false
			#pragma HLS DEPENDENCE variable=g1 inter false
			#pragma HLS DEPENDENCE variable=d1_temp_out inter false
			#pragma HLS DEPENDENCE variable=x_init inter false
			#pragma HLS DEPENDENCE variable=data_path1 inter false
			#pragma HLS DEPENDENCE variable=data_path2 inter false*/
				#pragma HLS unroll
			c_column_f_g1: for(int c=0;c<CLUSTERS;c++){
				#pragma HLS pipeline
				g1[r][0]=g1[r][0]+data_path2[c][r][0];
			}
			g1_g2_r[r][0]=g1[r][0]/g2[r][0].real();
			//g1_g2_r[r][0].real(g1[r][0].real()/g2[r][0].real());
			//g1_g2_r[r][0].imag(g1[r][0].imag()/g2[r][0].real());
			d1_temp_out[r][0]=x_init[r][0]-g1_g2_r[r][0];
			c_column_bcast: for(int c=0;c<CLUSTERS;c++){
				#pragma HLS unroll
				data_path1[c][r][0]=d1_temp_out[r][0];
			}



		}

}

void sqn_star_apex_cluster2(MATRIX_T g2[K][1],
		MATRIX_T data_path1[CLUSTERS][K][1],
		MATRIX_T data_path2[CLUSTERS][K][1],
		MATRIX_T x_init[K][1],
		int iter)

{
#pragma HLS function_instantiate variable=iter
//#pragma HLS INLINE RECURSIVE
		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
	    MATRIX_T g2_temp[K][1];
	    /*			#pragma HLS DEPENDENCE variable=g1 intra RAW true
	    			#pragma HLS DEPENDENCE variable=g1_g2_r intra RAW true
	    			#pragma HLS DEPENDENCE variable=d1_temp_out intra RAW true
	    			#pragma HLS DEPENDENCE variable=x_init intra WAR true
	    			#pragma HLS DEPENDENCE variable=g1_g2_r inter false
	    			#pragma HLS DEPENDENCE variable=g1 inter false
	    			#pragma HLS DEPENDENCE variable=d1_temp_out inter false
	    			#pragma HLS DEPENDENCE variable=x_init inter false*/


			#pragma HLS loop_flatten off
			b_row_loop_f_g1_1 :  for (int r=0;r<K;r++){
			#pragma HLS pipeline
				c_column_f_g1: for(int c=0;c<CLUSTERS;c++){
					#pragma HLS unroll
					g1[r][0]=g1[r][0]+data_path2[c][r][0];
				}
			}
				cmplx_division(g1,g2,g1_g2_r);
			    //cmplx_division_r(g1[r][0],g2[r][0],g1_g2_r[r][0]);
				/*b_row_loop_f_g1_2 :  for (int r=0;r<K;r++){
					#pragma HLS pipeline
					d1_temp_out[r][0]=x_init[r][0]-g1_g2_r[r][0];
					c_column_bcast: for(int c=0;c<CLUSTERS;c++){
						#pragma HLS unroll
						data_path1[c][r][0]=d1_temp_out[r][0];
					}
					x_init[r][0]=d1_temp_out[r][0];
				}*/


}

void sqn_star_apex_cluster3(MATRIX_T g2[K][1],
		MATRIX_T data_path1[CLUSTERS][K][1],
		MATRIX_T data_path2[CLUSTERS][K][1],
		MATRIX_T x_init[K][1],
		int iter)

{
#pragma HLS function_instantiate variable=iter
//#pragma HLS INLINE
#pragma HLS INLINE RECURSIVE
//#pragma HLS pipeline
		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
	    MATRIX_T g2_temp[K][1];


		#pragma HLS ARRAY_PARTITION variable=g1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g1_g2_r complete dim=1
		#pragma HLS ARRAY_PARTITION variable=x_init complete dim=1
		#pragma HLS ARRAY_PARTITION variable=d1_temp_out complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2

/*		if(iter==0){
			 b_row_loop_f_g2_1 : for(int r=0;r<K;r++){
				#pragma HLS pipeline
					c_column_f_g2_1: for(int c=0;c<CLUSTERS;c++){
						g2_temp[r][0]=g2_temp[r][0]+data_path1[c][r][0];
					}
				 d1_temp_in[r][0]=x_init[r][0];
				}
	     }*/


				c_column_f_g1: for(int c=0;c<CLUSTERS;c++){
			/*#pragma HLS DEPENDENCE variable=g1 intra RAW true
			#pragma HLS DEPENDENCE variable=g1_g2_r intra RAW true
			#pragma HLS DEPENDENCE variable=d1_temp_out intra RAW true
			#pragma HLS DEPENDENCE variable=x_init intra WAR true
			#pragma HLS DEPENDENCE variable=g1_g2_r inter false
			#pragma HLS DEPENDENCE variable=g1 inter false
			#pragma HLS DEPENDENCE variable=d1_temp_out inter false
			#pragma HLS DEPENDENCE variable=x_init inter false
			#pragma HLS DEPENDENCE variable=data_path1 inter false
			#pragma HLS DEPENDENCE variable=data_path2 inter false*/
				#pragma HLS pipeline
				b_row_loop_f_g1 :  for (int r=0;r<K;r++){
				#pragma HLS unroll
				g1[r][0]=g1[r][0]+data_path2[c][r][0];
			}
			}
			cmplx_division(g1,g2,g1_g2_r);

			b_row_loop_f_g1_3 :  for (int r=0;r<K;r++){

				#pragma HLS pipeline
				d1_temp_out[r][0]=x_init[r][0]-g1_g2_r[r][0];
				c_column_bcast: for(int c=0;c<CLUSTERS;c++){
				#pragma HLS unroll
				data_path1[c][r][0]=d1_temp_out[r][0];
			  }

			}



}

void sqn_star_frame(MATRIX_T y_in[M][1], MATRIX_T x_out[K][1])
{
	//works with 2 frames since dual port RAM
	#pragma HLS INTERFACE axis port=y_in
	#pragma HLS INTERFACE axis port=x_out
	MATRIX_T data_path1[MAX_ITERATION][CLUSTERS][K][1];
	MATRIX_T data_path2[MAX_ITERATION][CLUSTERS][K][1];
	MATRIX_T g2[K][1];
	MATRIX_T x_init[K][1];
	MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
	MATRIX_T clus_mat2[CLUSTERS][K][K];
	MATRIX_T y[CLUSTERS][ANTENNA_PER_CLUSTER][1];

	#pragma HLS RESOURCE variable=clus_mat1 core=RAM_2P
	#pragma HLS RESOURCE variable=clus_mat2 core=RAM_2P
	#pragma HLS RESOURCE variable=y core=RAM_2P
	#pragma HLS RESOURCE variable=data_path1 core=RAM_2P
	#pragma HLS RESOURCE variable=data_path2 core=RAM_2P
	#pragma HLS RESOURCE variable=g2 core=RAM_2P
	#pragma HLS RESOURCE variable=x_init core=RAM_2P



	#pragma HLS ARRAY_PARTITION variable=g2 complete
	#pragma HLS ARRAY_PARTITION variable=x_init complete
	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

	#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=y complete dim=1

	cl_l:for(int cls=0;cls<CLUSTERS;cls++){
			#pragma HLS unroll
		r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
					y[cls][r][0]=y_in[cls*ANTENNA_PER_CLUSTER+r][0];}

	#pragma HLS pipeline
	iterations_1: for(int it=0;it<MAX_ITERATION;it++)
		{
	#pragma HLS DEPENDENCE variable=data_path1 inter WAR true
	#pragma HLS DEPENDENCE variable=data_path2 inter WAR true
	#pragma HLS unroll
			clusters_1:for(int cl=0;cl<CLUSTERS;cl++)
			{
				#pragma HLS unroll
				sqn_star_cluster(clus_mat1[cl],clus_mat2[cl],y[cl],data_path1[it][cl],data_path2[it][cl],cl,it);
			if(cl==CLUSTERS-1){
			//	sqn_star_apex_cluster1(g2,data_path1,data_path2,x_init,it);
			//	sqn_star_apex_cluster(clus_mat1[cl],clus_mat2[cl],g2,y[cl],data_path1,data_path2,it);
				}
			}

		}

		r_l_x_st:for(int r=0;r<K;r++)
		#pragma HLS pipeline
		x_out[r][0]=data_path1[r][0];
}


void sqn_star_frame_throughput(MATRIX_T y[M][1], MATRIX_T x[K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

static MATRIX_T data_path1[CLUSTERS][K][1];
static MATRIX_T data_path2[CLUSTERS][K][1];
MATRIX_T g2[K][1];
MATRIX_T x_init[K][1];
MATRIX_T cluster_mat1[CLUSTERS][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS][K][K];
MATRIX_T yc[CLUSTERS][ANTENNA_PER_CLUSTER][1];



#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=g2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=x_init complete dim=1
/*
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=yc complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=g2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=x_init complete dim=2
*/
#pragma HLS RESOURCE variable=clus_mat1 core=RAM
	#pragma HLS RESOURCE variable=clus_mat2 core=RAM
	#pragma HLS RESOURCE variable=y core=RAM
	#pragma HLS RESOURCE variable=data_path1 core=RAM
	#pragma HLS RESOURCE variable=data_path2 core=RAM
	#pragma HLS RESOURCE variable=g2 core=RAM
	#pragma HLS RESOURCE variable=x_init core=RAM



//	frames_loop_1:for(int f=0;f<FRAMES;f++)
	//			#pragma HLS unroll
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[cls][r][0]=y[cls*ANTENNA_PER_CLUSTER+r][0];
//#pragma HLS pipeline
//frames_loop_2:for(int f=0;f<FRAMES;f++){
//#pragma HLS unroll
	/*		#pragma HLS DEPENDENCE variable=yc inter false
					#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
					#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
				//	#pragma HLS DEPENDENCE variable=data_path1 inter false
					//#pragma HLS DEPENDENCE variable=data_path2 inter false
					#pragma HLS DEPENDENCE variable=g2 inter false
					#pragma HLS DEPENDENCE variable=x_init inter false*/

	iterations_1: for(int it=0;it<MAX_ITERATION;it++)
		{
			#pragma HLS DEPENDENCE variable=yc inter false
				/*		#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
						#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
					//	#pragma HLS DEPENDENCE variable=data_path1 inter false
						//#pragma HLS DEPENDENCE variable=data_path2 inter false
						#pragma HLS DEPENDENCE variable=g2 inter false
						#pragma HLS DEPENDENCE variable=x_init inter false*/

//	#pragma HLS DEPENDENCE variable=data_path1 inter WAR true
	//#pragma HLS DEPENDENCE variable=data_path2 inter WAR true
	#pragma HLS unroll
			clusters_1:for(int cl=0;cl<CLUSTERS;cl++)
			{
				#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
//	#pragma HLS DEPENDENCE variable=data_path1 inter false
//#pragma HLS DEPENDENCE variable=data_path2 inter false
#pragma HLS DEPENDENCE variable=g2 inter false
#pragma HLS DEPENDENCE variable=x_init inter false

				#pragma HLS unroll
					sqn_star_cluster(cluster_mat1[cl],cluster_mat2[cl],yc[cl],data_path1[cl],data_path2[cl],it*CLUSTERS+cl,it);

			}
		sqn_star_apex_cluster3(g2,data_path1,data_path2,x_init,it);
			}

		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[r][0]=data_path1[CLUSTERS-1][r][0];
}


#if 0
void sqn_star_alg()
{
 MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
 MATRIX_T clus_mat2[CLUSTERS][K][K];
 MATRIX_T y[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][1];

//	MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
//	MATRIX_T clus_mat2[CLUSTERS][K][K];
//	MATRIX_T y[CLUSTERS][ANTENNA_PER_CLUSTER][1];
//	MATRIX_T g2[K][1];
//	MATRIX_T x_init[K][1];

//	MATRIX_T data_path1[CLUSTERS][K][1];
//	MATRIX_T data_path2[CLUSTERS][K][1];

//#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=y complete dim=1
//#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

//#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete
#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete
#pragma HLS ARRAY_PARTITION variable=y complete

//#pragma HLS DATAFLOW
for(int f=0;f<FRAMES;f++){
#pragma HLS unroll
//	sqn_star_frame(clus_mat1,clus_mat2,y[f]);

}
/*
	for(int it=0;it<MAX_ITERATION;it++)
	{

		for(int cl=0;cl<CLUSTERS;cl++)
		{
			#pragma HLS unroll
			sqn_star_cluster(clus_mat1[cl],clus_mat2[cl],y[cl],data_path1[cl],data_path2[cl],it);
		}
		sqn_star_apex_cluster1(g2,data_path1,data_path2,x_init,it);
	}*/
}

/////%%%%%%%%%%%%%%%%%%%%%%

void sqn_star_apex_cluster1_new(MATRIX_T g2[K][1],
		MATRIX_T data_path1[CLUSTERS][K][1],
		MATRIX_T data_path2[CLUSTERS][K][1],
		MATRIX_T x_init[K][1],
		int iter)

{
#pragma HLS dataflow
/*
#pragma HLS INTERFACE ap_none port=g2
#pragma HLS INTERFACE ap_none port=iter
#pragma HLS INTERFACE ap_none port=data_path1
#pragma HLS INTERFACE ap_none port=data_path2*/
#pragma HLS INLINE
//#pragma HLS pipeline
//#pragma HLS dataflow
//#pragma HLS function_instantiate variable=iter
//#pragma HLS INLINE RECURSIVE
		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
	    MATRIX_T g2_temp[K][1];
		if(iter==0){
			 b_row_loop_f_g2_1 : for(int r=0;r<K;r++){
				#pragma HLS pipeline
					c_column_f_g2_1: for(int c=0;c<CLUSTERS;c++){
						g2_temp[r][0]=g2_temp[r][0]+data_path1[c][r][0];
					}
				 d1_temp_in[r][0]=x_init[r][0];
				}
	     }

			b_row_loop_f_g1 :  for (int r=0;r<K;r++){
			#pragma HLS pipeline
			c_column_f_g1: for(int c=0;c<CLUSTERS;c++){
				#pragma HLS unroll
				g1[r][0]=g1[r][0]+data_path2[c][r][0];
			}
			/*
			g1_g2_r[r][0].real(g1[r][0].real()/g2[r][0].real());
			g1_g2_r[r][0].imag(g1[r][0].imag()/g2[r][0].real());*/
			cmplx_division(g1,g2,g1_g2_r);
			d1_temp_out[r][0]=d1_temp_in[r][0]-g1_g2_r[r][0];
			c_column_bcast: for(int c=0;c<CLUSTERS;c++){
				#pragma HLS unroll
				data_path1[c][r][0]=d1_temp_out[r][0];
			}



		}

}

void sqn_star_cluster_new(MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter)
{
MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K];
MATRIX_T cluster_mat2[K][K];

#pragma HLS INTERFACE ap_none port=yc
#pragma HLS INTERFACE ap_none port=clust_number
#pragma HLS INTERFACE ap_none port=iter
#pragma HLS INTERFACE ap_none port=data_path1
#pragma HLS INTERFACE ap_none port=data_path2

#pragma HLS dataflow

#pragma HLS function_instantiate variable=clust_number
#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];

    //#pragma HLS pipeline
	get_local_data_path:{
	#pragma HLS loop_merge
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS unroll
		d1_temp_in[r][0] = data_path1[r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS unroll
		d2_temp_in[r][0] = data_path2[r][0];
	}

	if(iter==0){
		f_gradient2(cluster_mat2,d1_temp_out);
	}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1(cluster_mat2,cluster_mat1,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS unroll
			d2_temp_out[r][0] = f_g1[r][0];
		}


		set_local_data_path:{
			#pragma HLS loop_merge
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS unroll
				 data_path1[r][0]=d1_temp_out[r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS unroll
				 data_path2[r][0]=d2_temp_out[r][0];
			}
}

void sqn_star_frame_new(MATRIX_T yin[M][1],MATRIX_T x_out[K][1]){//works with 2 frames since dual port RAM

#pragma HLS INTERFACE ap_none port=yin
#pragma HLS INTERFACE ap_none port=x_out

	MATRIX_T data_path1[CLUSTERS][K][1];
	MATRIX_T data_path2[CLUSTERS][K][1];
	MATRIX_T g2[K][1];
	MATRIX_T x_init[K][1];
	MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
	MATRIX_T clus_mat2[CLUSTERS][K][K];
	MATRIX_T y[CLUSTERS][ANTENNA_PER_CLUSTER][1];

#pragma HLS RESOURCE variable=clus_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=clus_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=y core=RAM_2P
#pragma HLS RESOURCE variable=data_path1 core=RAM_2P
#pragma HLS RESOURCE variable=data_path2 core=RAM_2P
#pragma HLS RESOURCE variable=g2 core=RAM_2P
#pragma HLS RESOURCE variable=x_init core=RAM_2P



//	#pragma HLS ARRAY_PARTITION variable=g2 complete
//	#pragma HLS ARRAY_PARTITION variable=x_init complete
	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete
	#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete
	#pragma HLS ARRAY_PARTITION variable=y complete


	cl_l:for(int cls=0;cls<CLUSTERS;cls++)
		r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
				#pragma HLS pipeline
			y[cls][r][0]=yin[cls*ANTENNA_PER_CLUSTER+r][0];

//#pragma HLS dataflow


	iterations_1: for(int it=0;it<MAX_ITERATION;it++)
		{
		#pragma HLS unroll
			clusters_1:for(int cl=0;cl<CLUSTERS;cl++)
			{
				#pragma HLS unroll
				#pragma HLS dependence variable=clus_mat1 inter false
				#pragma HLS dependence variable=clus_mat2 inter false
				#pragma HLS dependence variable=data_path1 inter false
				#pragma HLS dependence variable=data_path2 inter false
				#pragma HLS dependence variable=y inter false

				sqn_star_cluster_new(y[cl],data_path1[cl],data_path2[cl],it*CLUSTERS+cl,cl);
			if(cl==CLUSTERS-1){
					sqn_star_apex_cluster1_new(g2,data_path1,data_path2,x_init,it);
				}
			}
		}


	r_l_x_s:for(int r=0;r<K;r++)
		#pragma HLS pipeline
		x_out[r][0]=data_path1[0][r][0];



}

void sqn_star_frame2(MATRIX_T y_in[FRAMES][ANTENNA_PER_CLUSTER][1],MATRIX_T x_out[FRAMES][K][1]){//works with 2 frames since dual port RAM

#pragma HLS INTERFACE ap_none port=y_in
#pragma HLS INTERFACE ap_none port=x_out
	MATRIX_T data_path1[FRAMES][CLUSTERS][K][1];
	MATRIX_T data_path2[FRAMES][CLUSTERS][K][1];
	MATRIX_T g2[FRAMES][K][1];
	MATRIX_T x_init[FRAMES][K][1];
	MATRIX_T clus_mat1[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][K];
	MATRIX_T clus_mat2[FRAMES][CLUSTERS][K][K];
	MATRIX_T y[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][1];



#pragma HLS RESOURCE variable=clus_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=clus_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=y core=RAM_2P
#pragma HLS RESOURCE variable=data_path1 core=RAM_2P
#pragma HLS RESOURCE variable=data_path2 core=RAM_2P
#pragma HLS RESOURCE variable=g2 core=RAM_2P
#pragma HLS RESOURCE variable=x_init core=RAM_2P



	#pragma HLS ARRAY_PARTITION variable=g2 complete
	#pragma HLS ARRAY_PARTITION variable=x_init complete
	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2

	fr_st:for(int f=0;f<FRAMES;f++)
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS pipeline
				y[f][cls][r][0]=y_in[f][cls*CLUSTERS+r][0];

	#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=y complete dim=2

//#pragma HLS dataflow
//#pragma HLS pipeline
int f=0;
//#pragma HLS loop_merge
//frames:for(int f=0;f<FRAMES;f++){
	#pragma HLS pipeline
	iterations_1: for(int it=0;it<MAX_ITERATION;it++)
		{
	#pragma HLS unroll
			clusters_1:for(int cl=0;cl<CLUSTERS;cl++)
			{
				#pragma HLS unroll
				sqn_star_cluster(clus_mat1[f][cl],clus_mat2[f][cl],y[f][cl],data_path1[f][cl],data_path2[f][cl],it,cl);
			if(cl==CLUSTERS-1){
					sqn_star_apex_cluster1(g2[f],data_path1[f],data_path2[f],x_init[f],it);
				}
			}
		}
//	}
	fr_st1:for(int f=0;f<FRAMES;f++)
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS pipeline
			x_out[f][r][0]=data_path1[f][0][r][0];

}
#endif
void sqn_star_alg(MATRIX_T y1[FRAMES][M][1], MATRIX_T x1[FRAMES][K][1])
{

#pragma HLS INTERFACE axis port=y1
#pragma HLS INTERFACE axis port=x1
 //MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
 //MATRIX_T clus_mat2[CLUSTERS][K][K];
 MATRIX_T y[FRAMES][M][1];
 MATRIX_T x[FRAMES][K][1];

//	MATRIX_T clus_mat1[CLUSTERS][ANTENNA_PER_CLUSTER][K];
//	MATRIX_T clus_mat2[CLUSTERS][K][K];
//	MATRIX_T y[CLUSTERS][ANTENNA_PER_CLUSTER][1];
//	MATRIX_T g2[K][1];
//	MATRIX_T x_init[K][1];

//	MATRIX_T data_path1[CLUSTERS][K][1];
//	MATRIX_T data_path2[CLUSTERS][K][1];

//#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=y complete dim=1
//#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

//#pragma HLS ARRAY_PARTITION variable=clus_mat1 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=clus_mat2 complete dim=1
//#pragma HLS ARRAY_PARTITION variable=y complete dim=1

#pragma HLS ARRAY_PARTITION variable=y complete dim=1
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

 fr_sty:for(int f=0;f<FRAMES;f++)
		r_l_x_y:for(int r=0;r<M;r++)
			#pragma HLS pipeline
			y[f][r][0]=y1[f][r][0];

//#pragma HLS DATAFLOW
for(int f=0;f<FRAMES;f++){
	sqn_star_frame(y[f],x[f]);
}
	fr_st1:for(int f=0;f<FRAMES;f++)
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS pipeline
			x1[f][r][0]=x[f][r][0];

}

//############################################################

#if 0
void sqn_tree_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter)
{
#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];

	get_local_data_path:{
	#pragma HLS loop_merge
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d1_temp_in[r][0] = data_path1[r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d2_temp_in[r][0] = data_path2[r][0];

   	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];


	}

//	if(iter==0){
	//	f_gradient2(cluster_mat2,d1_temp_out);
	//}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			d2_temp_out[r][0] = f_g1[r][0];
		}


		set_local_data_path:{
			#pragma HLS loop_merge
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path1[r][0]=d1_temp_out[r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path2[r][0]=d2_temp_out[r][0];
			}
}

void sqn_tree_cluster_a1_branch(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[A1+1][K][1],
					MATRIX_T data_path2[A1+1][K][1],
					int clust_number,
					int iter)
{
#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[A1+1][K][1];
	MATRIX_T d2_temp_in[A1+1][K][1];
	MATRIX_T d1_temp_out[A1+1][K][1];
	MATRIX_T d2_temp_out[A1+1][K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];

	get_local_data_path:{
	#pragma HLS loop_merge
	c_column_unroll : for (int c=0;c<(A1+1);c++)
	{
		#pragma HLS UNROLL //try with pipeline
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d1_temp_in[c][r][0] = data_path1[c][r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d2_temp_in[c][r][0] = data_path2[c][r][0];

	}
   	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];


	}

//	if(iter==0){
	//	f_gradient2(cluster_mat2,d1_temp_out);
	//}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,d1_temp_in[0],f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			//add a loop here to accumulate the product.
			accumulate_f2:for(int a=1;a<A1+1;a++)
			{
				#pragma HLS PIPELINE
				d2_temp_out[0][r][0] = f_g1[r][0]+d2_temp_in[a][r][0];// this shoukd add f_g1[r][0]+d2_temp[1][r][0]+d2_temp
			}
		}
		bcast_x:for(int a=1;a<A1+1;a++)
		{
			#pragma HLS PIPELINE
			b_row_loop_f_g2 :  for (int r=0;r<B_ROWS_F;r++){
				d1_temp_out[a][r][0]=d1_temp_in[0][r][0];
			}
		}

		set_local_data_path:{
			#pragma HLS loop_merge
			c_column_out : for (int c=0;c<(A1+1);c++)
			{
			#pragma HLS UNROLL //try with unroll
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path1[c][r][0]=d1_temp_out[c][r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path2[c][r][0]=d2_temp_out[c][r][0];
			}
			}
}


//This is the apex cluster which is connected to only 2 clusters.
void sqn_tree_apex_cluster3(MATRIX_T g2[K][1],
		MATRIX_T data_path1[A1][K][1],
		MATRIX_T data_path2[A1][K][1],
		MATRIX_T x_init[K][1],
		int iter)

{
#pragma HLS function_instantiate variable=iter
//#pragma HLS INLINE
#pragma HLS INLINE RECURSIVE
//#pragma HLS pipeline
		MATRIX_T f_g2[K][1];
		MATRIX_T f_g1[K][1];
		MATRIX_T d1_temp_in[K][1];
		MATRIX_T d2_temp_in[K][1];
		MATRIX_T d1_temp_out[K][1];
		MATRIX_T d2_temp_out[K][1];
		MATRIX_T g1[K][1];
	    MATRIX_T g1_g2_r[K][1];
	    MATRIX_T g2_temp[K][1];


		#pragma HLS ARRAY_PARTITION variable=g1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g2 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=g1_g2_r complete dim=1
		#pragma HLS ARRAY_PARTITION variable=x_init complete dim=1
		#pragma HLS ARRAY_PARTITION variable=d1_temp_out complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
		#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2

/*		if(iter==0){
			 b_row_loop_f_g2_1 : for(int r=0;r<K;r++){
				#pragma HLS pipeline
					c_column_f_g2_1: for(int c=0;c<CLUSTERS;c++){
						g2_temp[r][0]=g2_temp[r][0]+data_path1[c][r][0];
					}
				 d1_temp_in[r][0]=x_init[r][0];
				}
	     }*/


				c_column_f_g1: for(int c=0;c<A1;c++){
			/*#pragma HLS DEPENDENCE variable=g1 intra RAW true
			#pragma HLS DEPENDENCE variable=g1_g2_r intra RAW true
			#pragma HLS DEPENDENCE variable=d1_temp_out intra RAW true
			#pragma HLS DEPENDENCE variable=x_init intra WAR true
			#pragma HLS DEPENDENCE variable=g1_g2_r inter false
			#pragma HLS DEPENDENCE variable=g1 inter false
			#pragma HLS DEPENDENCE variable=d1_temp_out inter false
			#pragma HLS DEPENDENCE variable=x_init inter false
			#pragma HLS DEPENDENCE variable=data_path1 inter false
			#pragma HLS DEPENDENCE variable=data_path2 inter false*/
				#pragma HLS pipeline
				b_row_loop_f_g1 :  for (int r=0;r<K;r++){
				#pragma HLS unroll
				g1[r][0]=g1[r][0]+data_path2[c][r][0];
			}
			}
			cmplx_division(g1,g2,g1_g2_r);

			b_row_loop_f_g1_3 :  for (int r=0;r<K;r++){

				#pragma HLS pipeline
				d1_temp_out[r][0]=x_init[r][0]-g1_g2_r[r][0];
				c_column_bcast: for(int c=0;c<A1;c++){
				#pragma HLS unroll
				data_path1[c][r][0]=d1_temp_out[r][0];
			  }

			}



}

void sqn_tree_frame_throughput(MATRIX_T y[M][1], MATRIX_T x[K][1])
{

	#pragma HLS INTERFACE ap_none port=y
	#pragma HLS INTERFACE ap_none port=x

	static MATRIX_T data_path1[CLUSTERS][K][1];
	static MATRIX_T data_path2[CLUSTERS][K][1];
	MATRIX_T g2[K][1];
	MATRIX_T x_init[K][1];
	MATRIX_T cluster_mat1[CLUSTERS][K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2[CLUSTERS][K][K];
	MATRIX_T yc[CLUSTERS][ANTENNA_PER_CLUSTER][1];



	#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=g2 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=x_init complete dim=1

    #pragma HLS RESOURCE variable=clus_mat1 core=RAM
	#pragma HLS RESOURCE variable=clus_mat2 core=RAM
	#pragma HLS RESOURCE variable=y core=RAM
	#pragma HLS RESOURCE variable=data_path1 core=RAM
	#pragma HLS RESOURCE variable=data_path2 core=RAM
	#pragma HLS RESOURCE variable=g2 core=RAM
	#pragma HLS RESOURCE variable=x_init core=RAM


		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
					#pragma HLS unroll
				yc[cls][r][0]=y[cls*CLUSTERS+r][0];
	iterations_1: for(int it=0;it<MAX_ITERATION;it++)
		{
			#pragma HLS DEPENDENCE variable=yc inter false
			#pragma HLS unroll
			clusters_1:for(int cl=0;cl<CLUSTERS;cl++)
			{
				#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
				#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
				#pragma HLS DEPENDENCE variable=g2 inter false
				#pragma HLS DEPENDENCE variable=x_init inter false
				#pragma HLS unroll

				if(cl>(CLUSTERS/2))
				if(cl>(3*CLUSTERS/4-1))

				sqn_tree_cluster(cluster_mat1[cl],cluster_mat2[cl],yc[cl],data_path1[cl],data_path2[cl],it*CLUSTERS+cl,it);

			}
		sqn_tree_apex_cluster3(g2,data_path1,data_path2,x_init,it);
			}


		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[r][0]=data_path1[CLUSTERS-1][r][0];
}

#endif

///////////////////////////////  Ring-STar topology /////////////////
#if 0
void sqn_ringstar_top(MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					MATRIX_T g2[K][1],
					int clust_number,
					MATRIX_T data_path1[S_CLUSTERS][K][1],
					MATRIX_T data_path2[S_CLUSTERS][K][1])

{

//#pragma HLS INLINE
#pragma HLS function_instantiate variable=clust_number
//#pragma HLS INLINE RECURSIVE
#pragma HLS pipeline


	//MATRIX_T g2[K][1];
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T g1[K][1];
	MATRIX_T g1_s[K][1];
    MATRIX_T g1_g2_r[K][1];

    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];


    	}
    			//implement apex cluster function here



    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS pipeline
    				    		    		data_path1_out_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_out_l[r][0] = 0;
    				    		}

    			}

    			else{
    			b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
									#pragma HLS unroll
				data_path2_out_l[r][0] = data_path2_in_l[r][0] + g1_s[r][0];
				data_path1_out_l[r][0]= data_path1_in_l[r][0];
				}

				c_column_f_g1: for(int c=0;c<S_CLUSTERS;c++){
				   					#pragma HLS pipeline
				   					b_row_loop_f_g1 :  for (int r=0;r<K;r++){
				    						#pragma HLS unroll
				    							g1_s[r][0]=g1_s[r][0]+data_path2[c][r][0]; //push g1 out
				    					}
				    				}
				b_row_loop_f_g1_4 :  for (int r=0;r<K;r++){
				 					#pragma HLS pipeline
				   					column_bcast: for(int c=0;c<S_CLUSTERS;c++){
				    							#pragma HLS unroll
				    							data_path1[c][r][0]=data_path1_in_l[r][0]; //send the value of x init obtained from previous computing block.
		    				 }
    				}
    			}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}

void sqn_ringstar_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number)
{
#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];

	get_local_data_path:{
	#pragma HLS loop_merge
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d1_temp_in[r][0] = data_path1[r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d2_temp_in[r][0] = data_path2[r][0];

   	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];


	}

//	if(iter==0){
	//	f_gradient2(cluster_mat2,d1_temp_out);
	//}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			d2_temp_out[r][0] = f_g1[r][0];
		}


		set_local_data_path:{
			#pragma HLS loop_merge
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path1[r][0]=d1_temp_out[r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path2[r][0]=d2_temp_out[r][0];
			}
}



void sqn_ringstar_frame_throughput(MATRIX_T y[FRAMES][M][1], MATRIX_T x[FRAMES][K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[FRAMES][CLUSTERS+1][S_CLUSTERS][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[FRAMES][CLUSTERS+1][S_CLUSTERS][K][K];
MATRIX_T yc[FRAMES][CLUSTERS+1][S_CLUSTERS][ANTENNA_PER_CLUSTER][1];
static MATRIX_T data_path1[FRAMES][CLUSTERS][S_CLUSTERS][K][1];
static MATRIX_T data_path2[FRAMES][CLUSTERS][S_CLUSTERS][K][1];


MATRIX_T g2_f1[FRAMES][K][1];
static MATRIX_T data_path1_in[FRAMES][CLUS1+2][K][1];
static MATRIX_T data_path2_in[FRAMES][CLUS1+2][K][1];

#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1

#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1
/*
//#pragma HLS ARRAY_PARTITION variable=yc complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2

#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2
//#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=3
//#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=3

*/

#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P

#pragma HLS RESOURCE variable=data_path1 core=RAM_2P
#pragma HLS RESOURCE variable=data_path2 core=RAM_2P



	frames_loop_1:for(int f=0;f<FRAMES;f++)
				#pragma HLS unroll
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			s_cl_l:for(int s_cls=0;s_cls<S_CLUSTERS;s_cls++)
					#pragma HLS unroll
					r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
							#pragma HLS unroll
						yc[f][cls][s_cls][r][0]=y[f][cls*ANTENNA_PER_CLUSTER+r][0]; //Need to change this cls*ANTENNA_PER_CLUSTER

frames_loop_2:for(int f=0;f<FRAMES;f++){
	#pragma HLS unroll
	#pragma HLS DEPENDENCE variable=yc inter false
	it_1: for(int it=0;it<CLUS1;it++)
	{
		#pragma HLS loop_flatten off
		#pragma HLS unroll
		#pragma HLS DEPENDENCE variable=yc inter false

		#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
		#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
		#pragma HLS DEPENDENCE variable=g2_f1 inter false
		#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
		#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

		sqn_ringstar_top(data_path1_in[f][it],data_path2_in[f][it],data_path1_in[f][it+1],data_path2_in[f][it+1],g2_f1[f],it,data_path1[f][it],data_path2[f][it]);

		if((it%(CLUSTERS+1))!=CLUSTERS){
		clusters_1:for(int cl=0;cl<S_CLUSTERS;cl++)
					{
						#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
						#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
						#pragma HLS DEPENDENCE variable=data_path1 inter false
						#pragma HLS DEPENDENCE variable=data_path2 inter false
						#pragma HLS unroll
						sqn_ringstar_cluster(cluster_mat1[f][it][cl],cluster_mat2[f][it][cl],yc[f][it][cl],data_path1[f][it][cl],data_path2[f][it][cl],cl);
					}
		}
	}
}
	frames_loop_3:for(int f=0;f<FRAMES;f++)
			#pragma HLS unroll
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[f][r][0]=data_path1_in[f][CLUS][r][0];
}

#endif

#if 0
void sqn_ringstar_throughput(MATRIX_T y[M][1], MATRIX_T x[K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[CLUSTERS+1][S_CLUSTERS][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS+1][S_CLUSTERS][K][K];
MATRIX_T yc[CLUSTERS+1][S_CLUSTERS][ANTENNA_PER_CLUSTER][1];
static MATRIX_T data_path1[CLUSTERS][S_CLUSTERS][K][1];
static MATRIX_T data_path2[CLUSTERS][S_CLUSTERS][K][1];


MATRIX_T g2_f1[K][1];
static MATRIX_T data_path1_in[CLUS1+2][K][1];
static MATRIX_T data_path2_in[CLUS1+2][K][1];

#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1

#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1

//#pragma HLS ARRAY_PARTITION variable=yc complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2

#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=2
//#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=3
//#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=3



#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P

#pragma HLS RESOURCE variable=data_path1 core=RAM_2P
#pragma HLS RESOURCE variable=data_path2 core=RAM_2P



	frames_loop_1:for(int f=0;f<FRAMES;f++)
				#pragma HLS unroll
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			s_cl_l:for(int s_cls=0;s_cls<S_CLUSTERS;s_cls++)
					#pragma HLS unroll
					r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
							#pragma HLS unroll
						yc[f][cls][s_cls][r][0]=y[f][cls*ANTENNA_PER_CLUSTER+r][0]; //Need to change this cls*ANTENNA_PER_CLUSTER

frames_loop_2:for(int f=0;f<FRAMES;f++){
	#pragma HLS unroll
	#pragma HLS DEPENDENCE variable=yc inter false
	it_1: for(int it=0;it<CLUS1;it++)
	{
		#pragma HLS loop_flatten off
		#pragma HLS unroll
		#pragma HLS DEPENDENCE variable=yc inter false

		#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
		#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
		#pragma HLS DEPENDENCE variable=g2_f1 inter false
		#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
		#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

		sqn_ringstar_cluster_flatten(data_path1_in[f][it],data_path2_in[f][it],data_path1_in[f][it+1],data_path2_in[f][it+1],g2_f1[f],it,data_path1[f][it],data_path2[f][it]);

		if((it%(CLUSTERS+1))!=CLUSTERS){
		clusters_1:for(int cl=0;cl<S_CLUSTERS;cl++)
					{
						#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
						#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
						#pragma HLS DEPENDENCE variable=data_path1 inter false
						#pragma HLS DEPENDENCE variable=data_path2 inter false
						#pragma HLS unroll
						sqn_ringstar_cluster(cluster_mat1[f][it][cl],cluster_mat2[f][it][cl],yc[f][it][cl],data_path1[f][it][cl],data_path2[f][it][cl],cl);
					}
		}
	}
}
	frames_loop_3:for(int f=0;f<FRAMES;f++)
			#pragma HLS unroll
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[f][r][0]=data_path1_in[f][CLUS][r][0];
}


#endif


//////Ringstar 2
void sqn_ringstar2_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number)
{
#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline
//#pragma HLS INLINE RECURSIVE
	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T d1_temp_in[K][1];
	MATRIX_T d2_temp_in[K][1];
	MATRIX_T d1_temp_out[K][1];
	MATRIX_T d2_temp_out[K][1];
	MATRIX_T g1[K][1];
    MATRIX_T g1_g2_r[K][1];
	MATRIX_T cluster_mat1_l[K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[K][K];


	get_local_data_path:{
	#pragma HLS loop_merge
	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d1_temp_in[r][0] = data_path1[r][0];

	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
		#pragma HLS PIPELINE
		d2_temp_in[r][0] = data_path2[r][0];

   	clust_mat1_l_r: for (int r=0;r<K;r++)
			#pragma HLS unroll
    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    			cluster_mat1_l[r][c]=cluster_mat1[r][c];

    clust_mat2_l_r: for (int r=0;r<K;r++)
    		#pragma HLS unroll
    		for(int c=0;c<K;c++)
    			cluster_mat2_l[r][c]=cluster_mat2[r][c];


	}

#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete
#pragma HLS ARRAY_PARTITION variable=yc complete
#pragma HLS ARRAY_PARTITION variable=d1_temp_in complete
#pragma HLS ARRAY_PARTITION variable=f_g1 complete
	f_gradient1_impr(cluster_mat2_l,cluster_mat1_l,yc,d1_temp_in,f_g1);
		b_row_loop_f_g1 :  for (int r=0;r<B_ROWS_F;r++){
			#pragma HLS pipeline
			d2_temp_out[r][0] = f_g1[r][0];
		}


		set_local_data_path:{
			#pragma HLS loop_merge
			b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path1[r][0]=d1_temp_out[r][0];

			b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
				#pragma HLS PIPELINE
				 data_path2[r][0]=d2_temp_out[r][0];
			}
}


void sqn_ringstar2_top(MATRIX_T cluster_mat1[S_CLUSTERS][K][ANTENNA_PER_CLUSTER],
					   MATRIX_T cluster_mat2[S_CLUSTERS][K][K],
					   MATRIX_T yc[S_CLUSTERS][ANTENNA_PER_CLUSTER][1],
		 			MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					MATRIX_T g2[K][1],
					int clust_number)
{

#pragma HLS function_instantiate variable=clust_number
#pragma HLS pipeline

	MATRIX_T f_g2[K][1];
	MATRIX_T f_g1[K][1];
	MATRIX_T data_path1_in_l[K][1];
	MATRIX_T data_path2_in_l[K][1];
	MATRIX_T data_path1_out_l[K][1];
	MATRIX_T data_path2_out_l[K][1];
	MATRIX_T g1[K][1];
	MATRIX_T g1_s[K][1];
    MATRIX_T g1_g2_r[K][1];
    static MATRIX_T data_path1[S_CLUSTERS][K][1];
   	static MATRIX_T data_path2[S_CLUSTERS][K][1];

	#pragma HLS ARRAY_PARTITION variable=data_path1 complete dim=1
	#pragma HLS ARRAY_PARTITION variable=data_path2 complete dim=1

   	MATRIX_T cluster_mat1_l[S_CLUSTERS][K][ANTENNA_PER_CLUSTER];
	MATRIX_T cluster_mat2_l[S_CLUSTERS][K][K];
	MATRIX_T yc_l[S_CLUSTERS][ANTENNA_PER_CLUSTER][1];

	#pragma HLS ARRAY_PARTITION variable=cluster_mat1_l complete dim=1
	#pragma HLS ARRAY_PARTITION variable=cluster_mat2_l complete dim=1
	#pragma HLS ARRAY_PARTITION variable=yc_l complete dim=1


    get_local_data_path:{
    	#pragma HLS loop_merge
    	b_row_loop_d1_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path1_in_l[r][0] = data_path1_in[r][0];

    	b_row_loop_d2_in :  for (int r=0;r<B_ROWS_F;r++)
    		#pragma HLS unroll
    		data_path2_in_l[r][0] = data_path2_in[r][0];

    	clust_mat1_l_r: for (int s_cl=0;s_cl<S_CLUSTERS;s_cl++)
    			#pragma HLS unroll
    	  	  for (int r=0;r<K;r++)
					#pragma HLS unroll
    	    		for(int c=0;c<ANTENNA_PER_CLUSTER;c++)
    	    			cluster_mat1_l[s_cl][r][c]=cluster_mat1[s_cl][r][c];

    	clust_mat2_l_r: for (int s_cl=0;s_cl<S_CLUSTERS;s_cl++)
    	    	#pragma HLS unroll
    		for (int r=0;r<K;r++)
				#pragma HLS unroll
    	    		for(int c=0;c<K;c++)
    	    			cluster_mat2_l[s_cl][r][c]=cluster_mat2[s_cl][r][c];

    	yc_l_r: for (int s_cl=0;s_cl<S_CLUSTERS;s_cl++)
    	    	   	#pragma HLS unroll
    	    		for (int r=0;r<ANTENNA_PER_CLUSTER;r++)
    	    	    	yc_l[s_cl][r][0]=yc[s_cl][r][0];


    	}
    			//implement apex cluster function here

    			if((clust_number%(CLUSTERS+1))==CLUSTERS){
    				cmplx_division(data_path2_in_l,g2,g1_g2_r);
    				row_loop_new:  for (int r=0;r<K;r++){
    				    		    		#pragma HLS pipeline
    				    		    		data_path1_out_l[r][0] = data_path1_in_l[r][0]-g1_g2_r[r][0];
    				    					data_path2_out_l[r][0] = 0;
    				    		}

    			}

    			else{
    				//ring star_cluster here

    				clusters_1:for(int s_cl=0;s_cl<S_CLUSTERS;s_cl++)
    				{
								#pragma HLS DEPENDENCE variable=cluster_mat1_l inter false
								#pragma HLS DEPENDENCE variable=cluster_mat2_l inter false
								#pragma HLS DEPENDENCE variable=yc_l inter false
								#pragma HLS DEPENDENCE variable=data_path1 inter false
								#pragma HLS DEPENDENCE variable=data_path2 inter false
								#pragma HLS unroll
								sqn_ringstar2_cluster(cluster_mat1_l[s_cl],cluster_mat2_l[s_cl],yc_l[s_cl],data_path1[s_cl],data_path2[s_cl],s_cl);
    				}

    			b_row_loop_f_g1_2 :  for (int r=0;r<B_ROWS_F;r++){
						#pragma HLS unroll
    								data_path2_out_l[r][0] = data_path2_in_l[r][0] + g1_s[r][0];
    								data_path1_out_l[r][0]= data_path1_in_l[r][0];
				}

				c_column_f_g1: for(int c=0;c<S_CLUSTERS;c++){
				   					#pragma HLS pipeline
				   					b_row_loop_f_g1 :  for (int r=0;r<K;r++){
				    						#pragma HLS unroll
				    							g1_s[r][0]=g1_s[r][0]+data_path2[c][r][0]; //push g1 out
				    					}
				    				}
				b_row_loop_f_g1_4 :  for (int r=0;r<K;r++){
				 					#pragma HLS pipeline
				   					column_bcast: for(int c=0;c<S_CLUSTERS;c++){
				    							#pragma HLS unroll
				    							data_path1[c][r][0]=data_path1_in_l[r][0]; //send the value of x init obtained from previous computing block.
		    				 }
    				}

    			}
		set_local_data_path:{
									#pragma HLS loop_merge
									b_row_loop_d1_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path1_out[r][0]=data_path1_out_l[r][0];

									b_row_loop_d2_out :  for (int r=0;r<B_ROWS_F;r++)
										#pragma HLS unroll
										 data_path2_out[r][0]=data_path2_out_l[r][0];
									}

}



void sqn_ringstar2_frame_throughput(MATRIX_T y[FRAMES][M][1], MATRIX_T x[FRAMES][K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[FRAMES][CLUSTERS+1][S_CLUSTERS][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[FRAMES][CLUSTERS+1][S_CLUSTERS][K][K];
MATRIX_T yc[FRAMES][CLUSTERS+1][S_CLUSTERS][ANTENNA_PER_CLUSTER][1];


MATRIX_T g2_f1[FRAMES][K][1];
static MATRIX_T data_path1_in[FRAMES][CLUS1+2][K][1];
static MATRIX_T data_path2_in[FRAMES][CLUS1+2][K][1];

#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1

#pragma HLS ARRAY_PARTITION variable=g2_f1 complete dim=1

#pragma HLS ARRAY_PARTITION variable=yc complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2



#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P




	frames_loop_1:for(int f=0;f<FRAMES;f++)
				#pragma HLS unroll
		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			s_cl_l:for(int s_cls=0;s_cls<S_CLUSTERS;s_cls++)
					#pragma HLS unroll
					r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
							#pragma HLS unroll
						yc[f][cls][s_cls][r][0]=y[f][(cls*(S_CLUSTERS*ANTENNA_PER_CLUSTER)+s_cls*ANTENNA_PER_CLUSTER)+r][0]; //Need to change this cls*ANTENNA_PER_CLUSTER

frames_loop_2:for(int f=0;f<FRAMES;f++){
	#pragma HLS unroll
	#pragma HLS DEPENDENCE variable=yc inter false
	it_1: for(int it=0;it<CLUS1;it++)
	{
		#pragma HLS loop_flatten off
		#pragma HLS unroll
		#pragma HLS DEPENDENCE variable=yc inter false

		#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
		#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
		#pragma HLS DEPENDENCE variable=g2_f1 inter false
		#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
		#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

		sqn_ringstar2_top(cluster_mat1[f][it],cluster_mat2[f][it],yc[f][it],data_path1_in[f][it],data_path2_in[f][it],data_path1_in[f][it+1],data_path2_in[f][it+1],g2_f1[f],it);


	}
}
	frames_loop_3:for(int f=0;f<FRAMES;f++)
			#pragma HLS unroll
		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[f][r][0]=data_path1_in[f][CLUS][r][0];
}

void sqn_ringstar2_throughput(MATRIX_T y[M][1], MATRIX_T x[K][1])
{


#pragma HLS INTERFACE ap_none port=y
#pragma HLS INTERFACE ap_none port=x

MATRIX_T cluster_mat1[CLUSTERS+1][S_CLUSTERS][K][ANTENNA_PER_CLUSTER];
MATRIX_T cluster_mat2[CLUSTERS+1][S_CLUSTERS][K][K];
MATRIX_T yc[CLUSTERS+1][S_CLUSTERS][ANTENNA_PER_CLUSTER][1];


MATRIX_T g2_f1[K][1];
static MATRIX_T data_path1_in[CLUS1+2][K][1];
static MATRIX_T data_path2_in[CLUS1+2][K][1];

/*#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=yc complete dim=1*/
#pragma HLS ARRAY_PARTITION variable=data_path1_in complete dim=1
#pragma HLS ARRAY_PARTITION variable=data_path2_in complete dim=1
/*
#pragma HLS ARRAY_PARTITION variable=cluster_mat1 complete dim=2
#pragma HLS ARRAY_PARTITION variable=cluster_mat2 complete dim=2
#pragma HLS ARRAY_PARTITION variable=yc complete dim=2*/


#pragma HLS RESOURCE variable=cluster_mat1 core=RAM_2P
#pragma HLS RESOURCE variable=cluster_mat2 core=RAM_2P
#pragma HLS RESOURCE variable=yc core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_in core=RAM_2P
#pragma HLS RESOURCE variable=data_path1_out core=RAM_2P
#pragma HLS RESOURCE variable=data_path2_out core=RAM_2P
#pragma HLS RESOURCE variable=g2_f1 core=RAM_2P




		cl_l:for(int cls=0;cls<CLUSTERS;cls++)
				#pragma HLS unroll
			s_cl_l:for(int s_cls=0;s_cls<S_CLUSTERS;s_cls++)
					#pragma HLS unroll
					r_l:for(int r=0;r<ANTENNA_PER_CLUSTER;r++)
							#pragma HLS unroll
						yc[cls][s_cls][r][0]=y[(cls*(S_CLUSTERS*ANTENNA_PER_CLUSTER)+s_cls*ANTENNA_PER_CLUSTER)+r][0]; //Need to change this cls*ANTENNA_PER_CLUSTER


	it_1: for(int it=0;it<CLUS1;it++)
	{
		#pragma HLS loop_flatten off
		#pragma HLS unroll
		#pragma HLS DEPENDENCE variable=yc inter false

		#pragma HLS DEPENDENCE variable=cluster_mat1 inter false
		#pragma HLS DEPENDENCE variable=cluster_mat2 inter false
		#pragma HLS DEPENDENCE variable=g2_f1 inter false
		#pragma HLS DEPENDENCE variable=data_path1_in inter RAW true
		#pragma HLS DEPENDENCE variable=data_path2_in inter RAW true

		sqn_ringstar2_top(cluster_mat1[it],cluster_mat2[it],yc[it],data_path1_in[it],data_path2_in[it],data_path1_in[it+1],data_path2_in[it+1],g2_f1,it);


	}

		r_l_x_s:for(int r=0;r<K;r++)
			#pragma HLS unroll
			x[r][0]=data_path1_in[CLUS][r][0];
}

