/* $Id: main.c,v 1.4 2004/04/21 04:23:43 pohlt Exp $ */

/*############################################################################*/

#include "main.h"
#include "lbm-brisbane.h"
#include <stdio.h>
#include <stdlib.h>

#if defined(SPEC)
#   include <time.h>
#else
#   include <sys/times.h>
#   include <unistd.h>
#endif

#include <sys/stat.h>

/*############################################################################*/

static LBM_GridPtr srcGrid, dstGrid;
size_t gridSize;
size_t marginSize;
double * src;
double * dst;

/*############################################################################*/

int main( int nArgs, char* arg[] ) {
    brisbane_init(&nArgs, &arg);
	MAIN_Param param;
	int t;

	MAIN_parseCommandLine( nArgs, arg, &param );
	MAIN_printInfo( &param );
	MAIN_initialize( &param );

  #pragma omp target data map(tofrom:src[0:gridSize]), map(to:dst[0:gridSize])
	{
	  for( t = 1; t <= param.nTimeSteps; t++ ) {
		  if( param.simType == CHANNEL ) {
			  LBM_handleInOutFlow( *srcGrid );
		  }
		  LBM_performStreamCollide( *srcGrid, *dstGrid );
		  LBM_swapGrids( &srcGrid, &dstGrid );
		  if( (t & 63) == 0 ) {
		    #pragma omp target update from(src[0:gridSize])
			  printf( "timestep: %i\n", t );
			  LBM_showGridStatistics( *srcGrid );
		  }
	  }
	} 
	MAIN_finalize( &param );

    brisbane_finalize();
	return 0;
}

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param ) {
	struct stat fileStat;
	
	int adjustArgs = 0;

        /* SPEC - handle one of --device/--platform */
        if ( nArgs == 8 ) adjustArgs+= 2;
        /* SPEC - handle both --device/--platform */
        if ( nArgs == 10 ) adjustArgs+= 4;
         
	if( nArgs < adjustArgs+5 || nArgs > adjustArgs+6 ) {
		printf( "syntax: lbm <time steps> <result file> <0: nil, 1: cmp, 2: str> <0: ldc, 1: channel flow> [<obstacle file>]\n" );
		exit( 1 );
	}

	param->nTimeSteps     = atoi( arg[adjustArgs+1] );
	param->resultFilename = arg[adjustArgs+2];
	param->action         = (MAIN_Action) atoi( arg[adjustArgs+3] );
	param->simType        = (MAIN_SimType) atoi( arg[adjustArgs+4] );

	if( nArgs == adjustArgs+6 ) {
		param->obstacleFilename = arg[adjustArgs+5];

		if( stat( param->obstacleFilename, &fileStat ) != 0 ) {
			printf( "MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
			         param->obstacleFilename );
			exit( 1 );
		}
		if( fileStat.st_size != SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z ) {
			printf( "MAIN_parseCommandLine:\n"
			        "\tsize of file '%s' is %i bytes\n"
					    "\texpected size is %i bytes\n",
			        param->obstacleFilename, (int) fileStat.st_size,
			        SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z );
			exit( 1 );
		}
	}
	else param->obstacleFilename = NULL;

	if( param->action == COMPARE &&
	    stat( param->resultFilename, &fileStat ) != 0 ) {
		printf( "MAIN_parseCommandLine: cannot stat result file '%s'\n",
		         param->resultFilename );
		exit( 1 );
	}
}

/*############################################################################*/

void MAIN_printInfo( const MAIN_Param* param ) {
	const char actionString[3][32] = {"nothing", "compare", "store"};
	const char simTypeString[3][32] = {"lid-driven cavity", "channel flow"};
	printf( "MAIN_printInfo:\n"
	        "\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
	        "\tnTimeSteps     : %i\n"
	        "\tresult file    : %s\n"
	        "\taction         : %s\n"
	        "\tsimulation type: %s\n"
	        "\tobstacle file  : %s\n\n",
	        SIZE_X, SIZE_Y, SIZE_Z, 1e-6*SIZE_X*SIZE_Y*SIZE_Z,
	        param->nTimeSteps, param->resultFilename, 
	        actionString[param->action], simTypeString[param->simType],
	        (param->obstacleFilename == NULL) ? "<none>" :
	                                            param->obstacleFilename );
}

/*############################################################################*/

void MAIN_initialize( const MAIN_Param* param) {
        

  LBM_allocateGrid( (double**) &srcGrid, (double**) &src );
  LBM_allocateGrid( (double**) &dstGrid, (double**) &dst );

	LBM_initializeGrid( *srcGrid );
	LBM_initializeGrid( *dstGrid );

	if( param->obstacleFilename != NULL ) {
		LBM_loadObstacleFile( *srcGrid, param->obstacleFilename );
		LBM_loadObstacleFile( *dstGrid, param->obstacleFilename );
	}

	if( param->simType == CHANNEL ) {
		LBM_initializeSpecialCellsForChannel( *srcGrid );
		LBM_initializeSpecialCellsForChannel( *dstGrid );
	}
	else {
		LBM_initializeSpecialCellsForLDC( *srcGrid );
		LBM_initializeSpecialCellsForLDC( *dstGrid );
	}

	LBM_showGridStatistics( *srcGrid );
}

/*############################################################################*/

void MAIN_finalize( const MAIN_Param* param ) {
	LBM_showGridStatistics( *srcGrid );

	if( param->action == COMPARE )
		LBM_compareVelocityField( *srcGrid, param->resultFilename, TRUE );
	if( param->action == STORE )
	LBM_storeVelocityField( *srcGrid, param->resultFilename, TRUE );

	LBM_freeGrid( (double**) &srcGrid );
	LBM_freeGrid( (double**) &dstGrid );
}

