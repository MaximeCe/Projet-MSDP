#          ms1.com
#          *********
gfortran -c ms2.f	      
gfortran -o ms1 ms1.f ms2.o  -L/usr/local/lib -lpgplot

