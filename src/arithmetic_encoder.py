#!/usr/bin/python3
# Script for adaptive arithmetic encoding

import arithmeticcoding
import pickle

class arithmetic_encoder(object):

    def compress(quantized, output_file):
        """
        Function to load d
        
        Input:
        filename : Input hdf5 file consisting of training dataset
        
        Output:
        dataframe of paths to images dataset
        """
        data = pickle.dumps(quantized)
        with open(output_file, "wb") as file:
            bitout = arithmeticcoding.BitOutputStream(file)
            initfreqs = arithmeticcoding.FlatFrequencyTable(257)
            freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
            enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
            i=0
            while i < len(data):
                # Read and encode one byte
                symbol = data[i]
                i += 1
                enc.write(freqs, symbol)
                freqs.increment(symbol)
            enc.write(freqs, 256)  # EOF
            enc.finish()  # Flush remaining code bits

    def decompress(input_file):
        decode = bytearray()
        with open(input_file, "rb") as inp:
            bitin = arithmeticcoding.BitInputStream(inp)
            initfreqs = arithmeticcoding.FlatFrequencyTable(257)
            freqs = arithmeticcoding.SimpleFrequencyTable(initfreqs)
            dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
            while True:
                # Decode and write one byte
                symbol = dec.read(freqs)
                if symbol == 256:  # EOF symbol
                    break
                decode.extend(bytes((symbol,)))
                freqs.increment(symbol)
        return pickle.loads(decode)
