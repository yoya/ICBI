for file in imgtest/bw_0*256.bmp
do
./release/icbi $file 
convert -flip pippo.pgm $file.tif
done

