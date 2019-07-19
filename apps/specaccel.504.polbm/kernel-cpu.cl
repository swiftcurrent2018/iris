#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define SIZE   (100)
#define SIZE_X (1*SIZE)
#define SIZE_Y (1*SIZE)
#define SIZE_Z (130)

#define OMEGA (1.95)

#define DFL1 (1.0/ 3.0)
#define DFL2 (1.0/18.0)
#define DFL3 (1.0/36.0)

typedef enum {C = 0,
              N, S, E, W, T, B,
              NE, NW, SE, SW,
              NT, NB, ST, SB,
              ET, EB, WT, WB,
              FLAGS, N_CELL_ENTRIES} CELL_ENTRIES;

typedef enum {OBSTACLE    = 1 << 0,
              ACCEL       = 1 << 1,
              IN_OUT_FLOW = 1 << 2} CELL_FLAGS;

#define CALC_INDEX(x,y,z,e) ((e)+N_CELL_ENTRIES*((x)+ \
                             (y)*SIZE_X+(z)*SIZE_X*SIZE_Y))

#define SWEEP_VAR int i;

#define SWEEP_START(x1,y1,z1,x2,y2,z2) \
	for( i = CALC_INDEX(x1, y1, z1, 0); \
	     i < CALC_INDEX(x2, y2, z2, 0); \
			 i += N_CELL_ENTRIES ) {

#define SWEEP_END }

#define SWEEP_X  ((i / N_CELL_ENTRIES) % SIZE_X)
#define SWEEP_Y (((i / N_CELL_ENTRIES) / SIZE_X) % SIZE_Y)
#define SWEEP_Z  ((i / N_CELL_ENTRIES) / (SIZE_X*SIZE_Y))

#define GRID_ENTRY(g,x,y,z,e)          ((g)[CALC_INDEX( x,  y,  z, e)])
#define GRID_ENTRY_SWEEP(g,dx,dy,dz,e) ((g)[CALC_INDEX(dx, dy, dz, e)+(i)])

#define LOCAL(g,e)       (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_C(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_N(g,e)  (GRID_ENTRY_SWEEP( g,  0, +1,  0, e ))
#define NEIGHBOR_S(g,e)  (GRID_ENTRY_SWEEP( g,  0, -1,  0, e ))
#define NEIGHBOR_E(g,e)  (GRID_ENTRY_SWEEP( g, +1,  0,  0, e ))
#define NEIGHBOR_W(g,e)  (GRID_ENTRY_SWEEP( g, -1,  0,  0, e ))
#define NEIGHBOR_T(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, +1, e ))
#define NEIGHBOR_B(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, -1, e ))
#define NEIGHBOR_NE(g,e) (GRID_ENTRY_SWEEP( g, +1, +1,  0, e ))
#define NEIGHBOR_NW(g,e) (GRID_ENTRY_SWEEP( g, -1, +1,  0, e ))
#define NEIGHBOR_SE(g,e) (GRID_ENTRY_SWEEP( g, +1, -1,  0, e ))
#define NEIGHBOR_SW(g,e) (GRID_ENTRY_SWEEP( g, -1, -1,  0, e ))
#define NEIGHBOR_NT(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, +1, e ))
#define NEIGHBOR_NB(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, -1, e ))
#define NEIGHBOR_ST(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, +1, e ))
#define NEIGHBOR_SB(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, -1, e ))
#define NEIGHBOR_ET(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, +1, e ))
#define NEIGHBOR_EB(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, -1, e ))
#define NEIGHBOR_WT(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, +1, e ))
#define NEIGHBOR_WB(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, -1, e ))


#define SRC_C(g)  (LOCAL( g, C  ))
#define SRC_N(g)  (LOCAL( g, N  ))
#define SRC_S(g)  (LOCAL( g, S  ))
#define SRC_E(g)  (LOCAL( g, E  ))
#define SRC_W(g)  (LOCAL( g, W  ))
#define SRC_T(g)  (LOCAL( g, T  ))
#define SRC_B(g)  (LOCAL( g, B  ))
#define SRC_NE(g) (LOCAL( g, NE ))
#define SRC_NW(g) (LOCAL( g, NW ))
#define SRC_SE(g) (LOCAL( g, SE ))
#define SRC_SW(g) (LOCAL( g, SW ))
#define SRC_NT(g) (LOCAL( g, NT ))
#define SRC_NB(g) (LOCAL( g, NB ))
#define SRC_ST(g) (LOCAL( g, ST ))
#define SRC_SB(g) (LOCAL( g, SB ))
#define SRC_ET(g) (LOCAL( g, ET ))
#define SRC_EB(g) (LOCAL( g, EB ))
#define SRC_WT(g) (LOCAL( g, WT ))
#define SRC_WB(g) (LOCAL( g, WB ))

#define DST_C(g)  (NEIGHBOR_C ( g, C  ))
#define DST_N(g)  (NEIGHBOR_N ( g, N  ))
#define DST_S(g)  (NEIGHBOR_S ( g, S  ))
#define DST_E(g)  (NEIGHBOR_E ( g, E  ))
#define DST_W(g)  (NEIGHBOR_W ( g, W  ))
#define DST_T(g)  (NEIGHBOR_T ( g, T  ))
#define DST_B(g)  (NEIGHBOR_B ( g, B  ))
#define DST_NE(g) (NEIGHBOR_NE( g, NE ))
#define DST_NW(g) (NEIGHBOR_NW( g, NW ))
#define DST_SE(g) (NEIGHBOR_SE( g, SE ))
#define DST_SW(g) (NEIGHBOR_SW( g, SW ))
#define DST_NT(g) (NEIGHBOR_NT( g, NT ))
#define DST_NB(g) (NEIGHBOR_NB( g, NB ))
#define DST_ST(g) (NEIGHBOR_ST( g, ST ))
#define DST_SB(g) (NEIGHBOR_SB( g, SB ))
#define DST_ET(g) (NEIGHBOR_ET( g, ET ))
#define DST_EB(g) (NEIGHBOR_EB( g, EB ))
#define DST_WT(g) (NEIGHBOR_WT( g, WT ))
#define DST_WB(g) (NEIGHBOR_WB( g, WB ))

#define MAGIC_CAST(v) ((unsigned int*) ((void*) (&(v))))
#define FLAG_VAR(v) unsigned int* const _aux_ = MAGIC_CAST(v)

#define TEST_FLAG_SWEEP(g,f)     ((*MAGIC_CAST(LOCAL(g, FLAGS))) & (f))
#define SET_FLAG_SWEEP(g,f)      {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG_SWEEP(g,f)    {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS_SWEEP(g) {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_)  =    0;}

#define TEST_FLAG(g,x,y,z,f)     ((*MAGIC_CAST(GRID_ENTRY(g, x, y, z, FLAGS))) & (f))
#define SET_FLAG(g,x,y,z,f)      {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG(g,x,y,z,f)    {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS(g,x,y,z) {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_)  =    0;}

__kernel void hIOF_0(__global double* restrict src) {
    int i = get_global_id(0) * N_CELL_ENTRIES;

	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;
    __global double* srcGrid = src + margin;

    double ux, uy, uz, rho, ux1, uy1, uz1, rho1, ux2, uy2, uz2, rho2, u2, px, py;

    rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );
    rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

    rho = 2.0*rho1 - rho2;

    px = (SWEEP_X / (0.5*(SIZE_X-1))) - 1.0;
    py = (SWEEP_Y / (0.5*(SIZE_Y-1))) - 1.0;
    ux = 0.00;
    uy = 0.00;
    uz = 0.01 * (1.0-px*px) * (1.0-py*py);

    u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

    LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

    LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
    LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
    LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
    LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
    LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
    LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

    LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
    LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
    LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
    LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
    LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
    LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
    LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
    LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
    LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
    LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
    LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
    LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
}

__kernel void hIOF_1(__global double* restrict src) {
    int i = get_global_id(0) * N_CELL_ENTRIES;

	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;
    __global double* srcGrid = src + margin;

    double ux, uy, uz, rho, ux1, uy1, uz1, rho1, ux2, uy2, uz2, rho2, u2, px, py;

    rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
    ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
    uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
    uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

    ux1 /= rho1;
    uy1 /= rho1;
    uz1 /= rho1;

    rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
    ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
    uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
        - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
    uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
        + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

    ux2 /= rho2;
    uy2 /= rho2;
    uz2 /= rho2;

    rho = 1.0;

    ux = 2*ux1 - ux2;
    uy = 2*uy1 - uy2;
    uz = 2*uz1 - uz2;

    u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

    LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

    LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
    LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
    LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
    LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
    LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
    LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

    LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
    LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
    LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
    LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
    LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
    LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
    LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
    LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
    LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
    LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
    LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
    LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
}

__kernel void pSC(__global double* restrict dst, __global double* restrict src) {
    int i = get_global_id(0) * N_CELL_ENTRIES;

	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

    __global double* dstGrid = dst + margin;
    __global double* srcGrid = src + margin;

    double ux, uy, uz, u2, rho;
    int test_flag_sweep;

    test_flag_sweep = (unsigned int) LOCAL(srcGrid, FLAGS);
    if (test_flag_sweep & OBSTACLE) {
//    if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
        DST_C ( dstGrid ) = SRC_C ( srcGrid );
        DST_S ( dstGrid ) = SRC_N ( srcGrid );
        DST_N ( dstGrid ) = SRC_S ( srcGrid );
        DST_W ( dstGrid ) = SRC_E ( srcGrid );
        DST_E ( dstGrid ) = SRC_W ( srcGrid );
        DST_B ( dstGrid ) = SRC_T ( srcGrid );
        DST_T ( dstGrid ) = SRC_B ( srcGrid );
        DST_SW( dstGrid ) = SRC_NE( srcGrid );
        DST_SE( dstGrid ) = SRC_NW( srcGrid );
        DST_NW( dstGrid ) = SRC_SE( srcGrid );
        DST_NE( dstGrid ) = SRC_SW( srcGrid );
        DST_SB( dstGrid ) = SRC_NT( srcGrid );
        DST_ST( dstGrid ) = SRC_NB( srcGrid );
        DST_NB( dstGrid ) = SRC_ST( srcGrid );
        DST_NT( dstGrid ) = SRC_SB( srcGrid );
        DST_WB( dstGrid ) = SRC_ET( srcGrid );
        DST_WT( dstGrid ) = SRC_EB( srcGrid );
        DST_EB( dstGrid ) = SRC_WT( srcGrid );
        DST_ET( dstGrid ) = SRC_WB( srcGrid );
        return;
    }

    rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
        + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
        + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
        + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
        + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
        + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
        + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
        + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
        + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
        + SRC_WB( srcGrid );
    ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
        + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
        + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
        + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
        - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
    uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
        + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
        - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
        + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
        - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
    uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
        + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
        + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
        + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
        + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

    ux /= rho;
    uy /= rho;
    uz /= rho;

    test_flag_sweep  = (unsigned int) LOCAL(srcGrid, FLAGS);
    if (test_flag_sweep & ACCEL) {
    //if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
        ux = 0.005;
        uy = 0.002;
        uz = 0.000;
    }

    u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

    DST_C ( dstGrid ) = (1.0-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0                                 - u2);
    DST_N ( dstGrid ) = (1.0-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
    DST_S ( dstGrid ) = (1.0-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
    DST_E ( dstGrid ) = (1.0-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
    DST_W ( dstGrid ) = (1.0-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
    DST_T ( dstGrid ) = (1.0-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
    DST_B ( dstGrid ) = (1.0-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

    DST_NE( dstGrid ) = (1.0-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
    DST_NW( dstGrid ) = (1.0-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
    DST_SE( dstGrid ) = (1.0-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
    DST_SW( dstGrid ) = (1.0-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
    DST_NT( dstGrid ) = (1.0-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
    DST_NB( dstGrid ) = (1.0-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
    DST_ST( dstGrid ) = (1.0-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
    DST_SB( dstGrid ) = (1.0-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
    DST_ET( dstGrid ) = (1.0-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
    DST_EB( dstGrid ) = (1.0-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
    DST_WT( dstGrid ) = (1.0-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
    DST_WB( dstGrid ) = (1.0-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
}
