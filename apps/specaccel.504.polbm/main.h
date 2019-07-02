/* $Id: main.h,v 1.4 2004/04/21 04:23:43 pohlt Exp $ */

/*############################################################################*/

#ifndef _MAIN_H_
#define _MAIN_H_

/*############################################################################*/

#include "config.h"

/*############################################################################*/

typedef enum {NOTHING = 0, COMPARE, STORE} MAIN_Action;
typedef enum {LDC = 0, CHANNEL} MAIN_SimType;

typedef struct {
	int nTimeSteps;
	char* resultFilename;
	MAIN_Action action;
	MAIN_SimType simType;
	char* obstacleFilename;
} MAIN_Param;

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param );
void MAIN_printInfo( const MAIN_Param* param );
void MAIN_initialize( const MAIN_Param* param );
void MAIN_finalize( const MAIN_Param* param );

/*############################################################################*/

#endif /* _MAIN_H_ */
