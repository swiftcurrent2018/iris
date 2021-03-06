#ifndef BRISBANE_SRC_RT_DEBUG_H
#define BRISBANE_SRC_RT_DEBUG_H

#include <stdio.h>
#include <string.h>

#define _TRACE_ENABLE
#define _CHECK_ENABLE
#define _DEBUG_ENABLE
#define _INFO_ENABLE
#define _ERROR_ENABLE
#define _TODO_ENABLE
#define _CLERROR_ENABLE
#define _CUERROR_ENABLE
#define _HIPERROR_ENABLE

#define _COLOR_DEBUG

#ifdef _COLOR_DEBUG
#define RED     "\033[22;31m"
#define GREEN   "\033[22;32m"
#define YELLOW  "\033[22;33m"
#define BLUE    "\033[22;34m"
#define PURPLE  "\033[22;35m"
#define CYAN    "\033[22;36m"
#define GRAY    "\033[22;37m"
#define BRED    "\033[1;31m"
#define BGREEN  "\033[1;32m"
#define BYELLOW "\033[1;33m"
#define BBLUE   "\033[1;34m"
#define BPURPLE "\033[1;35m"
#define BCYAN   "\033[1;36m"
#define BGRAY   "\033[1;37m"
#define _RED    "\033[22;41m" BGRAY
#define _GREEN  "\033[22;42m" BGRAY
#define _YELLOW "\033[22;43m" BGRAY
#define _BLUE   "\033[22;44m" BGRAY
#define _PURPLE "\033[22;45m" BGRAY
#define _CYAN   "\033[22;46m" BGRAY
#define _GRAY   "\033[22;47m"
#define RESET   "\x1b[m"
#else
#define RED
#define GREEN
#define YELLOW
#define BLUE
#define PURPLE
#define CYAN
#define GRAY
#define BRED
#define BGREEN
#define BYELLOW
#define BBLUE
#define BPURPLE
#define BCYAN
#define BGRAY
#define _RED
#define _GREEN
#define _YELLOW
#define _BLUE
#define _PURPLE
#define _CYAN
#define _GRAY
#define RESET
#endif

#define CHECK_O   "\u2714 "
#define CHECK_X   "\u2716 "

namespace brisbane {
namespace rt {

extern char brisbane_log_prefix_[];

#define __SHORT_FILE__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef _TRACE_ENABLE
#define  _trace(fmt, ...) { printf( BLUE "[T] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#define __trace(fmt, ...) { printf(_BLUE "[T] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define  _trace(fmt, ...)
#define __trace(fmt, ...)
#endif

#ifdef _CHECK_ENABLE
#define  _check() { printf( PURPLE "[C] %s [%s:%d:%s]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__); fflush(stdout); }
#define __check() { printf(_PURPLE "[C] %s [%s:%d:%s]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__); fflush(stdout); }
#else
#define  _check()
#define __check()
#endif

#ifdef _DEBUG_ENABLE
#define  _debug(fmt, ...) { printf( CYAN "[D] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#define __debug(fmt, ...) { printf(_CYAN "[D] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define  _debug(fmt, ...)
#define __debug(fmt, ...)
#endif

#ifdef _INFO_ENABLE
#define  _info(fmt, ...) { printf( YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#define __info(fmt, ...) { printf(_YELLOW "[I] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define  _info(fmt, ...)
#define __info(fmt, ...)
#endif

#ifdef _ERROR_ENABLE
#define  _error(fmt, ...) { printf( RED "[E] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#define __error(fmt, ...) { printf(_RED "[E] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define  _error(fmt, ...)
#define __error(fmt, ...)
#endif

#ifdef _TODO_ENABLE
#define  _todo(fmt, ...) { printf( GREEN "[TODO] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#define __todo(fmt, ...) { printf(_GREEN "[TODO] %s [%s:%d:%s] " fmt RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, __VA_ARGS__); fflush(stdout); }
#else
#define  _todo(fmt, ...)
#define __todo(fmt, ...)
#endif

#ifdef _CLERROR_ENABLE
#define  _clerror(err) { if (err != CL_SUCCESS) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#define __clerror(err) { if (err != CL_SUCCESS) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#else
#define  _clerror(err)
#define __clerror(err)
#endif

#ifdef _CUERROR_ENABLE
#define  _cuerror(err) { if (err != CUDA_SUCCESS) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#define __cuerror(err) { if (err != CUDA_SUCCESS) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#else
#define  _clerror(err)
#define __clerror(err)
#endif

#ifdef _HIPERROR_ENABLE
#define  _hiperror(err) { if (err != hipSuccess) { printf( RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#define __hiperror(err) { if (err != hipSuccess) { printf(_RED "[E] %s [%s:%d:%s] err[%d]" RESET "\n", brisbane_log_prefix_, __SHORT_FILE__, __LINE__, __func__, err); fflush(stdout); } }
#else
#define  _hiperror(err)
#define __hiperror(err)
#endif

} /* namespace rt */
} /* namespace brisbane */

#endif /* BRISBANE_SRC_RT_DEBUG_H */
