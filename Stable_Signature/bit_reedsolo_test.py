from bit_reedsolo import  ReedSolomonCodec_BW, UncorrectableError_RS
import random
# --- Example Usage ---
if __name__ == '__main__':
    print("--- Reed-Solomon (Binary List Interface) with Berlekamp-Welch Example ---")
    try:
        # RS Code parameters: k=3 message symbols, 2 parity symbols (so n=7).
        # Using m=3 for GF(2^3). Primitive poly for GF(2^3) is x^3+x+1 (0b1011 = 11)
        # t = 2 // 2 = 1 symbol errors can be corrected.
        # for 48 bits, (n,m) = (16,3) (12,4), (8,6), (6,8), (4,12)
        # (k, k*m, parity, m) = (11, 44 ,1, 4) ()
        k_rs, num_parity_rs, m_rs = 5, 2, 8
        
        rs_codec = ReedSolomonCodec_BW(k_rs, num_parity_rs, m_rs)
        print(f"RS Codec: n={rs_codec.n_sym}, k={rs_codec.k_sym}, m={rs_codec.m}, t_correctable={rs_codec.t_correctable}")
        print(f"Using GF(2^{m_rs}) with primitive polynomial 0x{rs_codec.gf.primitive_poly:x}")

        # Message needs to be k*m bits long, as a list
        message_bit_list = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        print(f"Original Message Bit List: {message_bit_list}")

        encoded_bit_list = rs_codec.encode(message_bit_list)
        print(f"Encoded Codeword Bit List: {encoded_bit_list} (n*m = {rs_codec.n_sym * rs_codec.m} bits)")

        # No error
        print("\nDecoding (No Error):")
        try:
            msg_bits, corrected_cw_bits, num_corr = rs_codec.decode(list(encoded_bit_list)) # Pass a copy
            print(f"  Decoded Message Bits: {msg_bits}")
            # print(f"  Corrected Codeword Bits: {corrected_cw_bits}") # Can be long
            print(f"  Symbol Errors Corrected: {num_corr}")
            print(f"  Message matches original: {msg_bits == message_bit_list}")
            print(f"  Codeword matches original: {corrected_cw_bits == encoded_bit_list}")
        except UncorrectableError_RS as e:
            print(f"  Error: {e}")

        # 1-bit error (within one symbol)
        print("\nDecoding (1 Bit Error -> 1 Symbol Error):")
        received_1bit_err_list = list(encoded_bit_list)
        err_bit_idx = 7 # e.g., change 8th bit (index 7)
        if err_bit_idx < len(received_1bit_err_list):
            received_1bit_err_list[err_bit_idx] = 1 - received_1bit_err_list[err_bit_idx] # Flip bit
            print(f"  Corrupted Bit List (1 bit err at index {err_bit_idx}): {received_1bit_err_list}")
            try:
                msg_bits, corrected_cw_bits, num_corr = rs_codec.decode(received_1bit_err_list)
                print(f"  Decoded Message Bits: {msg_bits}")
                print(f"  Symbol Errors Corrected: {num_corr}")
                print(f"  Message matches original: {msg_bits == message_bit_list}")
                print(f"  Codeword matches original: {corrected_cw_bits == encoded_bit_list}")
            except UncorrectableError_RS as e:
                print(f"  Error: {e}")
        
        # 2-symbol errors (affecting multiple bits)
        print("\nDecoding (Multiple Bit Errors -> 2 Symbol Errors):")
        received_2sym_err_list = list(encoded_bit_list)
        err_bit_idx_a = 7 # Affects symbol 1 (indices 3,4,5)
        err_bit_idx_b = 1 # Affects symbol 4 (indices 12,13,14)
        if err_bit_idx_a < len(received_2sym_err_list) and err_bit_idx_b < len(received_2sym_err_list):
            received_2sym_err_list[err_bit_idx_a] = 1 - received_2sym_err_list[err_bit_idx_a]
            received_2sym_err_list[err_bit_idx_b] = 1 - received_2sym_err_list[err_bit_idx_b]
            print(f"  Corrupted Bit List (2 bit err at indices {err_bit_idx_a},{err_bit_idx_b}): {received_2sym_err_list}")
            try:
                msg_bits, corrected_cw_bits, num_corr = rs_codec.decode(received_2sym_err_list)
                print(f"  Decoded Message Bits: {msg_bits}")
                print(f"  Symbol Errors Corrected: {num_corr}")
                print(f"  Message matches original: {msg_bits == message_bit_list}")
                print(f"  Codeword matches original: {corrected_cw_bits == encoded_bit_list}")
            except UncorrectableError_RS as e:
                print(f"  Error: {e}")

        # 3-symbol errors (affecting multiple bits, should be uncorrectable for t=2)
        print("\nDecoding (Multiple Bit Errors -> 3 Symbol Errors - Uncorrectable):")
        received_3sym_err_list = list(encoded_bit_list)
        err_indices = [1, 8, 16] # Affects symbols 0, 2, 5
        possible_to_corrupt = True
        for idx in err_indices:
             if idx >= len(received_3sym_err_list): possible_to_corrupt = False; break
             received_3sym_err_list[idx] = 1 - received_3sym_err_list[idx]
        
        if possible_to_corrupt:
            print(f"  Corrupted Bit List (3 bit err at indices {err_indices}): {received_3sym_err_list}")
            try:
                rs_codec.decode(received_3sym_err_list)
            except UncorrectableError_RS as e:
                print(f"  Caught Expected Error: {e}")
        else:
             print("  Skipping 3-error test due to index out of bounds.")
        
        print("\n--- Test Case: 48-bit message (k=6, m=8), 4 parity symbols (t=2) ---")
        try:
            k_48, num_parity_48, m_48 = 6, 4, 8 # n=10, k=6, t=2
            codec_48 = ReedSolomonCodec_BW(k_48, num_parity_48, m_48)
            print(f"RS Codec: n={codec_48.n_sym}, k={codec_48.k_sym}, m={codec_48.m}, t_correctable={codec_48.t_correctable}")
            print(f"Using GF(2^{m_48}) with primitive polynomial 0x{codec_48.gf.primitive_poly:x}")

            # Create a 48-bit message list (k*m = 6*8 = 48)
            message_48_list = [random.randint(0,1) for _ in range(k_48 * m_48)]
            print(f"Original 48-bit Message List: {message_48_list}")

            encoded_48_list = codec_48.encode(message_48_list)
            print(f"Encoded Bit List (n*m = {codec_48.n_sym * codec_48.m} bits): {encoded_48_list}")

            # Test with 2 symbol errors (e.g., corrupt bits in symbol 1 and symbol 5)
            corrupted_48_list = list(encoded_48_list)
            err_bit_idx_48a = 10 # Affects symbol 1 (bits 8-15)
            err_bit_idx_48b = 42 # Affects symbol 5 (bits 40-47)
            corrupted_48_list[err_bit_idx_48a] = 1 - corrupted_48_list[err_bit_idx_48a]
            corrupted_48_list[err_bit_idx_48b] = 1 - corrupted_48_list[err_bit_idx_48b]
            print(f"Corrupted (2 symbol errors via bits {err_bit_idx_48a},{err_bit_idx_48b}): {corrupted_48_list}")

            msg_bits_48, corrected_cw_bits_48, num_corr_48 = codec_48.decode(corrupted_48_list)
            print(f"  Decoded Message Bits: {msg_bits_48}")
            print(f"  Symbol Errors Corrected: {num_corr_48}")
            print(f"  Message matches original: {msg_bits_48 == message_48_list}")

        except ValueError as ve:
            print(f"ValueError during 48-bit test: {ve}")
        except UncorrectableError_RS as e:
            print(f"UncorrectableError during 48-bit test: {e}")


    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")
        import traceback
        traceback.print_exc()