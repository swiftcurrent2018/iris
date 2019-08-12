LIB=libLLVMBrisbane.so

$(LIB):Brisbane.cpp.o
	g++ -fPIC  -fPIC -fvisibility-inlines-hidden -Werror=date-time -std=c++11 -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -pedantic -Wno-long-long -Wimplicit-fallthrough -Wno-maybe-uninitialized -Wno-class-memaccess -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wno-comment -g  -Wl,-rpath-link,/home/eck/work/llvm-project/build/./lib -Wl,-z,nodelete -shared  -o $@ $^ -Wl,-rpath,"\$ORIGIN/../lib"
	#g++ -fPIC  -fPIC -fvisibility-inlines-hidden -Werror=date-time -std=c++11 -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -pedantic -Wno-long-long -Wimplicit-fallthrough -Wno-maybe-uninitialized -Wno-class-memaccess -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wno-comment -g  -Wl,-rpath-link,/home/eck/work/llvm-project/build/./lib   -Wl,--version-script,"LLVMBrisbane.exports"  -Wl,-z,nodelete -shared  -o $@ $^ -Wl,-rpath,"\$ORIGIN/../lib"

Brisbane.cpp.o:Brisbane.cpp
	g++  -DGTEST_HAS_RTTI=0 -D_DEBUG -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -I$(HOME)/install/include -I$(HOME)/install/include/polly -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -fPIC -fvisibility-inlines-hidden -Werror=date-time -std=c++11 -Wall -Wextra -Wno-unused-parameter -Wwrite-strings -Wcast-qual -Wno-missing-field-initializers -pedantic -Wno-long-long -Wimplicit-fallthrough -Wno-maybe-uninitialized -Wno-class-memaccess -Wno-noexcept-type -Wdelete-non-virtual-dtor -Wno-comment -g -fPIC    -fno-exceptions -fno-rtti -o $@ -c $^
	#clang -g3 -shared -fPIC -o $@ $^ -D_GLIBCXX_USE_CXX11_ABI=0 -fno-rtti -Wl,-headerpad_max_install_names -undefined dynamic_lookup -I ~/install/include/ -I ~/install/include/polly/

install:$(LIB)
	mv $^ $(HOME)/install/lib

clean:
	rm -f $(LIB) Brisbane.cpp.o


