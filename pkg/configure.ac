
AC_INIT([mvtnorm], 0.9-9992) dnl package name, version

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
    echo "could not determine R_HOME"
    exit 1
fi
                             
CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`

dnl needed for x64 with at least one gcc version
PKG_CFLAGS=""
if test  "${CC}" = "gcc -std=gnu99" ; then
    PKG_CFLAGS="-O2 -ffloat-store"
fi;

AC_SUBST(PKG_CFLAGS)                                  

AC_CONFIG_FILES([src/Makevars])

AC_OUTPUT
