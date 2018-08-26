%for verify
load('JpegCoeff.mat');
num = 0;
DC_vector = randi(2047,3,1);
DC_code = DC_coeff(DC_vector);
DC_code2 = binstr2array(DC_code);
answer = DC_decoder(DC_code2,DCTAB)
    
AC_vector = randi(2047,63,1)
AC_code = AC_coeff(AC_vector)
AC_code2 = binstr2array(AC_code);
AC2 = AC_decoder(AC_code2,ACTAB)

