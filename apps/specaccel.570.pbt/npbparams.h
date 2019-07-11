/* CLASS = W */
/*
   This file is generated automatically by the setparams utility.
   It sets the number of processors and the class of the NPB
   in this directory. Do not modify it by hand.   
*/
/*
known issue: can not change the macros to regular variables here 
to use the read inputdata.bt mode. Thus we need to manually change 
the PROBLEM_SIZE and NITER_DEFAUT to run different sizes.
*/
/*
test size(CLASS W in NAS)
    #define PROBLEM_SIZE 24
    #define DT_DEFAULT 0.0008
train size(CLASS A in NAS)
    #define PROBLEM_SIZE 64
    #define DT_DEFAULT 0.0008
ref size(CLASS B in NAS)
    #define PROBLEM_SIZE 102
    #define DT_DEFAULT 0.0003
*/
#define PROBLEM_SIZE 102
#define NITER_DEFAULT 200
#define DT_DEFAULT 0.0003
#define CONVERTDOUBLE  false
#define COMPILETIME "23 Apr 2013"
#define NPBVERSION "3.3.1"
#define CS1 "icc"
#define CS2 "icc"
#define CS3 "-lm"
#define CS4 "-I../common"
#define CS5 "-O3 -mcmodel=medium"
#define CS6 "-O3 -mcmodel=medium"
#define CS7 "randdp"
