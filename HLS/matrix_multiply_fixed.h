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


void sqn_star_frame_new(MATRIX_T y[FRAMES][CLUSTERS][ANTENNA_PER_CLUSTER][1]);




#endif

