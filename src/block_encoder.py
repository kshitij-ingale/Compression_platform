class block_encoder(object):

    def compress(data, out_file = 'compressed_x'):
    """
    Function to write quantized vector of image as a binary file
    
    Input:
    data : Quantized vector of image
    out_file : File name for binary file to be stored
    
    Output:
    None (File saved in location)
    """
    
    bitstring = ''            
    for center in data.astype(int).flatten():
        if center == 0:
            bitstring+='00'
        elif center == 1:
            bitstring+='10'
        elif center == 2:
            bitstring+='01'
        elif center == 3:
            bitstring += '11'
        else:
            print('Error in encoding' )
        
    print('File size = ',len(bitstring))
    with open(out_file+'.bin', 'wb') as f:
        f.write(bitstring)

    def decompress(input_file):
        """
        Function to write quantized vector of image as a binary file
        
        Input:
        input_file : File name for binary file to be restored to reconstructed image
        
        Output:
        Matrix containing quantized vector for reconstruction
        """

        with open(input_file, 'rb') as f:
            data=str(f.read())
        f.close()

        comp = []
        for i in range(0,len(data),2):
            bit = data[i:i+2]
            if bit == '00':
                comp.append(0.)
            elif bit == '10':
                comp.append(1.)
            elif bit == '01':
                comp.append(2.)
            elif bit == '11':
                comp.append(3.)
        im_mat = np.array(comp)
        return im_mat
