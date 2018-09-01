function pic = JPEG_decoder(jpegcodes)
pic = picture_recover(binstr2array(jpegcodes.DC_code),binstr2array(jpegcodes.AC_code),jpegcodes.H,jpegcodes.W);
end