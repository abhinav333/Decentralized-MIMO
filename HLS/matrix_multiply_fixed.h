/*****************************************************************************
 *
 *     Author: Xilinx, Inc.
 *
 *     This text contains proprietary, confidential information of
 *     Xilinx, Inc. , is distributed by under license from Xilinx,
 *     Inc., and may be used, copied and/or disclosed only pursuant to
 *     the terms of a valid license agreement with Xilinx, Inc.
 *
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
 *
 *     Xilinx products are not intended for use in life support appliances,
 *     devices, or systems. Use in such applications is expressly prohibited.
 *
 *     (c) Copyright 2013-2014 Xilinx Inc.
 *     All rights reserved.
 *
 *****************************************************************************/

#ifndef MATRIX_MULTIPLY_H
#define MATRIX_MULTIPLY_H

#include "hls_linear_algebra.h"
#include "hls_stream.h"

typedef ap_fixed<16,8> CMPLX_TYPE;
//typedef float CMPLX_TYPE;

const unsigned CLUSTERS=4;	//number of CLUSTERS
const unsigned S_CLUSTERS=8;
const unsigned ANTENNA_PER_CLUSTER = 16; //number of users
const unsigned M=CLUSTERS*S_CLUSTERS*ANTENNA_PER_CLUSTER;     //number of antennas
const unsigned K=4;	  //number of users
const unsigned MAX_ITERATION=3; //algorithm iterations
const unsigned ALPHA = 1;
const unsigned BETTA = 1;
const unsigned FRAMES=1;
const unsigned A1=2;


#define CLUS MAX_ITERATION*CLUSTERS
#define CLUS1 (CLUSTERS+1)*MAX_ITERATION


const unsigned A_ROWS = 3;
const unsigned A_COLS = 3;
const unsigned B_ROWS = 3;
const unsigned B_COLS = 1;
const unsigned C_ROWS = 3;
const unsigned C_COLS = 1;

const unsigned A_ROWS_F = K;
const unsigned A_COLS_F = K;
const unsigned B_ROWS_F = K;
const unsigned B_COLS_F = 1;
const unsigned C_ROWS_F = K;
const unsigned C_COLS_F = 1;

// Define implementation type
// typedef ap_fixed<16,8> MATRIX_T;


typedef hls::x_complex<CMPLX_TYPE > MATRIX_T;




typedef hls::x_traits<MATRIX_T,MATRIX_T>::MADD_T TEST;
typedef hls::x_complex<TEST > MATRIX_T2;
typedef hls::stream<MATRIX_T > MATRIX_T_STREAM;
//typedef hls::x_complex<MULT_RT, CMPLX_TYPE > MATRIX_T_D;
//typedef hls::x_complex<float> MATRIX_T;
/*
// Define configuration class for each implementation "target"
struct MULT_CONFIG_SMALL: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,A_ROWS,A_COLS,B_ROWS,B_COLS,MATRIX_T, MATRIX_T>{
  static const int INNER_II = 4;
};
struct MULT_CONFIG_FAST: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,A_ROWS,A_COLS,B_ROWS,B_COLS,MATRIX_T, MATRIX_T>{
  static const int ARCH = 2;
  static const int UNROLL_FACTOR = 4;
};
struct MULT_CONFIG_FASTER: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,A_ROWS,A_COLS,B_ROWS,B_COLS,MATRIX_T, MATRIX_T>{
  static const int ARCH = 4;
};
*/
struct MULT_CONFIG_FAST: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,A_ROWS,A_COLS,B_ROWS,B_COLS,MATRIX_T, MATRIX_T>{
  static const int ARCH = 2;
  static const int UNROLL_FACTOR = 4;
};
struct MULT_CONFIG_FASTER: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,A_ROWS,A_COLS,B_ROWS,B_COLS,MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;
};

struct MULT_CONFIG_FASTER_f_g1_1: hls::matrix_multiply_traits<hls::NoTranspose,hls::NoTranspose,K,K,K,1,MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;

};

struct MULT_CONFIG_FASTER_f_g1_2: hls::matrix_multiply_traits<hls::ConjugateTranspose, hls::NoTranspose,ANTENNA_PER_CLUSTER,K,ANTENNA_PER_CLUSTER, 1, MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;
};

struct MULT_CONFIG_FASTER_f_g1_3: hls::matrix_multiply_traits<hls::NoTranspose, hls::NoTranspose,1,K,K,1, MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;

};

struct MULT_CONFIG_FASTER_f_g1_4: hls::matrix_multiply_traits<hls::NoTranspose, hls::NoTranspose,1,ANTENNA_PER_CLUSTER,ANTENNA_PER_CLUSTER, 1, MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;

};

struct MULT_CONFIG_FASTER_dcd1: hls::matrix_multiply_traits<hls::ConjugateTranspose, hls::NoTranspose,ANTENNA_PER_CLUSTER,1,ANTENNA_PER_CLUSTER, 1, MATRIX_T, MATRIX_T>{
  static const int ARCH = 0;
};


void matrix_multiply_default(const MATRIX_T A [A_ROWS][A_COLS],
                             const MATRIX_T B [B_ROWS][B_COLS],
                                   MATRIX_T C [C_ROWS][C_COLS],int& inverse_OK);

void forward_substituion(MATRIX_T a [A_ROWS_F][A_COLS_F],
                         MATRIX_T y [B_ROWS_F][B_COLS_F],
                         MATRIX_T x [C_ROWS_F][C_COLS_F]);

void adm_cluster(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
				 MATRIX_T cluster_mat2[K][K],
				 MATRIX_T cluster_mat3[K][K],
				 MATRIX_T lambdac[K][1],
				 MATRIX_T zc[K][1],
				 MATRIX_T y[ANTENNA_PER_CLUSTER][1],
				 MATRIX_T x_init[K][1],
				 MATRIX_T x_est[K][1],
				 MATRIX_T cluster_number,
				 MATRIX_T alpha,
				 MATRIX_T betta,
				 MATRIX_T vc[K][1]);
void adm_frame(MATRIX_T cluster_mat1[CLUSTERS][K][ANTENNA_PER_CLUSTER],
		 MATRIX_T cluster_mat2[CLUSTERS][K][K],
		 MATRIX_T cluster_mat3[CLUSTERS][K][K],
		 MATRIX_T y[CLUSTERS][ANTENNA_PER_CLUSTER][1],
		 MATRIX_T x_est[CLUSTERS][K][1]);
void dcd_frame(MATRIX_T y_c[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][1],
MATRIX_T x_est[FRAMES][K][1]);
void dcd_cluster(MATRIX_T muc[K][1],
			MATRIX_T nuc[K][1],
			MATRIX_T cluster_mat1_cd[ANTENNA_PER_CLUSTER][K],
			MATRIX_T y_c[ANTENNA_PER_CLUSTER][1],
			MATRIX_T x_cluster_estimate[K][1],int cl);
void dcd_acc_core(MATRIX_T x_cluster_estimate[CLUSTERS][K][1],MATRIX_T x[K][1]);
//			void dcd_algorithm(MATRIX_T a [A_ROWS_F][A_COLS_F],
  //                       MATRIX_T y [B_ROWS_F][B_COLS_F],
    //                               MATRIX_T x [C_ROWS_F][C_COLS_F]);


void sqn_ring_cluster_flatten_system_gen(MATRIX_T cluster_mat1[K][ANTENNA_PER_CLUSTER],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1_in[K][1],
					MATRIX_T data_path2_in[K][1],
					MATRIX_T data_path1_out[K][1],
					MATRIX_T data_path2_out[K][1],
					int clust_number);



template <int W2, int IW2, int W1, int IW1>
void fxp_sqrt(ap_fixed<W2,IW2>& result, ap_fixed<W1,IW1>& in_val)
{
   enum { QW = (IW1+1)/2 + (W2-IW2) + 1 }; // derive max root width
   enum { SCALE = (W2 - W1) - (IW2 - (IW1+1)/2) }; // scale (shift) to adj initial remainer value
   enum { ROOT_PREC = QW - (IW1 % 2) };
   assert((IW1+1)/2 <= IW2); // Check that output format can accommodate full result

   ap_int<QW> q      = 0;   // partial sqrt
   ap_int<QW> q_star = 0;   // diminished partial sqrt
   ap_int<QW+2> s; // scaled remainder initialized to extracted input bits
   if (SCALE >= 0)
      s = in_val.range(W1-1,0) << (SCALE);
   else
      s = ((in_val.range(W1-1,0) >> (0 - (SCALE + 1))) + 1) >> 1;

   // Non-restoring square-root algorithm
   for (int i = 0; i <= ROOT_PREC; i++) {
      if (s >= 0) {
         s = 2 * s - (((ap_int<QW+2>(q) << 2) | 1) << (ROOT_PREC - i));
         q_star = q << 1;
         q = (q << 1) | 1;
      } else {
         s = 2 * s + (((ap_int<QW+2>(q_star) << 2) | 3) << (ROOT_PREC - i));
         q = (q_star << 1) | 1;
         q_star <<= 1;
      }
   }
   // Round result by "extra iteration" method
   if (s > 0)
      q = q + 1;
   // Truncate excess bit and assign to output format
   result.range(W2-1,0) = ap_int<W2>(q >> 1);
}


template<int R1,int C1, int R2, int C2> //Normal x Normal
void mat_multiply(MATRIX_T a[R1][C1],MATRIX_T b[R2][C2],MATRIX_T c[R1][C2])
{
	a_row_loop_m :  for (int row=0;row<R1;row++){
			b_col_loop:for (int col=0;col<C2;col++){
				m_loop:for (int m=0;m<C1;m++){
					#pragma HLS unroll
					c[row][col]=c[row][col]+(a[row][m]*b[m][col]);
				}
			}
	}
}


template<int R1,int C1, int R2, int C2> //Conj X Normal
void mat_multiply1(MATRIX_T a[R1][C1],MATRIX_T b[R2][C2],MATRIX_T c[R1][C2])
{
	a_row_loop_m :  for (int row=0;row<R1;row++){
			b_col_loop:for (int col=0;col<C2;col++){
				m_loop:for (int m=0;m<C1;m++){
					#pragma HLS unroll
					c[row][col]=c[row][col]+ (x_conj(a[m][row])*b[m][col]);
				}
			}
	}
}


template<int R1,int C1, int R2, int C2> //Normal X Conj
void mat_multiply2(MATRIX_T a[R1][C1],MATRIX_T b[R2][C2],MATRIX_T c[R1][C2])
{
	a_row_loop_m :  for (int row=0;row<R1;row++){
			b_col_loop:for (int col=0;col<C2;col++){
				m_loop:for (int m=0;m<C1;m++){
					#pragma HLS unroll
					c[row][col]=c[row][col]+ (a[row][m]*x_conj(b[col][m]));
				}
			}
	}
}


void f_gradient2(MATRIX_T cluster_mat2[K][K],
					MATRIX_T f2[K][1]);

void f_gradient1(MATRIX_T cluster_mat2[K][K],
				MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K],
				MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
				MATRIX_T x_e[K][1],
				MATRIX_T f1[K][1]);

void sqn_ring_cluster(MATRIX_T g2[K][1],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter);


void sqn_star_cluster(MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K],
					MATRIX_T cluster_mat2[K][K],
					MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
					MATRIX_T data_path1[K][1],
					MATRIX_T data_path2[K][1],
					int clust_number,
					int iter);


void sqn_star_apex_cluster(MATRIX_T cluster_mat1[ANTENNA_PER_CLUSTER][K],
		MATRIX_T cluster_mat2[K][K],
		MATRIX_T g2[K][1],
		MATRIX_T yc[ANTENNA_PER_CLUSTER][1],
		MATRIX_T data_path1[CLUSTERS-1][K][1],
		MATRIX_T data_path2[CLUSTERS-1][K][1],
		int iter);

void sqn_star_apex_cluster1(MATRIX_T g2[K][1],
		MATRIX_T data_path1[CLUSTERS][K][1],
		MATRIX_T data_path2[CLUSTERS][K][1],
		MATRIX_T x_init[K][1],
		int iter);

void sqn_star_alg();
void sqn_ring_alg();

void sqn_star_frame_new(MATRIX_T y[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][1]);




#endif

