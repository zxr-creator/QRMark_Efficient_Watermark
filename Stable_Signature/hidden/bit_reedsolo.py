import numpy as np # For potential future use, not strictly needed for this version
import math
import random # For generating random messages and errors
from typing import List, Tuple

class UncorrectableError_RS(Exception):
    """Custom exception for Reed-Solomon decoding errors."""
    def __init__(self, message, received_input=None, syndrome_str=None): # syndrome_str might not be available/relevant for BW
        super().__init__(message)
        self.received_input = received_input # Could be binary list or symbols
        self.syndrome_str = syndrome_str # May be None
    def __str__(self):
        details = ""
        # Truncate long inputs for display
        input_str = str(self.received_input)
        if isinstance(self.received_input, list) and len(self.received_input) > 100:
            input_str = str(self.received_input[:100]) + "...]"

        if self.received_input is not None:
            details = f"Received Input: {input_str}"
        # if self.syndrome_str: # Syndromes are internal to BW solving process
        #     details += f", Syndromes: {self.syndrome_str}"
        return f"{super().__str__()} - {details}"

class GF2m:
    """Represents elements and performs arithmetic in a Galois Field GF(2^m)."""
    def __init__(self, m, primitive_poly_int):
        if m <= 0: raise ValueError("Field dimension m must be positive.")
        self.m = m
        self.order = 1 << m
        self.primitive_poly = primitive_poly_int

        # Check if the primitive polynomial degree matches m
        # The degree of a polynomial represented as an int is its bit_length() - 1
        # The primitive polynomial itself should have degree m, so its int representation includes m+1 bits (e.g., x^4+x+1 is 10011, bit_length 5 for m=4)
        if primitive_poly_int > 0 and (primitive_poly_int.bit_length() -1) != m :
            # Special case for m=1, poly=0b11 (x+1), bit_length is 2. (2-1)=1. This is correct.
            # The check should be if the highest bit is at position m.
             if not (m==1 and primitive_poly_int == 0b11): # x+1 for GF(2)
                print(f"Warning: Primitive polynomial 0x{primitive_poly_int:x} (degree {primitive_poly_int.bit_length()-1}) might not be appropriate for m={m}.")

        self.alpha_to = [0] * self.order # Stores alpha^i at index i
        self.index_of = [-1] * self.order # Stores i at index alpha^i (log base alpha)
        
        val = 1 # Start with alpha^0 = 1
        for i in range(self.order - 1): # Iterate for alpha^0 to alpha^(order-2)
            self.alpha_to[i] = val
            if val < self.order : # Ensure val is within field bounds before indexing
                 self.index_of[val] = i
            val <<= 1 # Multiply by alpha (which is x, or 2 in polynomial basis)
            if val >= self.order: # If degree is m or more, reduce by primitive polynomial
                val ^= self.primitive_poly
        # self.alpha_to[self.order - 1] will remain 0, which is fine as it's not used for alpha^(order-1) = 1
        # self.index_of[0] remains -1 (log of 0 is undefined)

    def add(self, a, b): return a ^ b
    def subtract(self, a, b): return a ^ b # In GF(2^m), addition is subtraction

    def multiply(self, a, b):
        if a == 0 or b == 0: return 0
        if not (0 <= a < self.order and 0 <= b < self.order):
            raise ValueError(f"Inputs {a},{b} out of range for GF(2^{self.m})")
        log_a = self.index_of[a]
        log_b = self.index_of[b]
        if log_a == -1 or log_b == -1: # Should not happen if a,b !=0
            raise ValueError(f"Log of non-zero element not found: a={a} (log={log_a}), b={b} (log={log_b}) in GF(2^{self.m})")
        
        # Sum of logs modulo (order-1) because alpha^(order-1) = 1
        log_sum = (log_a + log_b) % (self.order - 1)
        return self.alpha_to[log_sum]

    def power(self, a, n):
        if not (0 <= a < self.order):
            raise ValueError(f"Base {a} out of range for GF(2^{self.m})")
        if a == 0: return 0 if n > 0 else 1 # 0^0 = 1, 0^n = 0 for n>0
        if n == 0: return 1 # a^0 = 1 for a != 0
        
        log_a = self.index_of[a]
        if log_a == -1: raise ValueError(f"Log of non-zero element {a} not found for power.")
        
        # Exponent should be modulo (order-1)
        exp_val = n % (self.order - 1)
        # Handle negative exponents if n was negative, (n % k + k) % k ensures positive result
        if exp_val < 0: exp_val += (self.order - 1) 
            
        log_res = (log_a * exp_val) % (self.order - 1)
        return self.alpha_to[log_res]

    def inverse(self, a):
        if not (0 < a < self.order): 
            raise ValueError(f"Input {a} invalid or zero for inverse in GF(2^{self.m})")
        log_a = self.index_of[a]
        if log_a == -1: raise ValueError(f"Log of non-zero element {a} not found for inverse.")
        
        if self.order - 1 == 0: return 1 # GF(2^1) if m=0, order = 1 (not typical) or m=1, order-1 = 1
                                         # For m=1, order=2, order-1=1. log_inv = (1-log_a)%1 = 0. alpha_to[0]=1. Correct.

        # Inverse of a = a^(order-2). log(a_inv) = log_a * (order-2) % (order-1)
        # Or, a * a_inv = 1 => log_a + log_a_inv = 0 mod (order-1)
        # => log_a_inv = -log_a mod (order-1) = (order-1 - log_a) mod (order-1)
        log_inv = (self.order - 1 - log_a) % (self.order - 1) 
        return self.alpha_to[log_inv]

class GF2mPolynomial:
    """Polynomials with coefficients in GF(2^m). Coeffs: [c0, c1, ..., ck] for c0 + c1*x + ..."""
    def __init__(self, gf_field: GF2m, coeffs_gf2m: list):
        self.gf = gf_field
        # Validate coefficients
        for c in coeffs_gf2m:
            if not (0 <= c < self.gf.order):
                raise ValueError(f"Coefficient {c} out of range for GF(2^{self.gf.m})")
        # Ensure coeffs is a new list, handle empty list case
        self.coeffs = list(coeffs_gf2m) if coeffs_gf2m else [0] 
        self._remove_leading_zeros()

    def _remove_leading_zeros(self):
        """Removes leading zero coefficients (highest powers) unless it's the zero polynomial."""
        while len(self.coeffs) > 1 and self.coeffs[-1] == 0:
            self.coeffs.pop()
        if not self.coeffs: # Should not happen if constructor ensures [0] for empty
            self.coeffs.append(0)

    def degree(self):
        """Returns the degree of the polynomial. Degree of 0 is -1."""
        if len(self.coeffs) == 1 and self.coeffs[0] == 0: return -1 # Zero polynomial
        return len(self.coeffs) - 1

    def is_zero(self): return self.degree() == -1

    def __add__(self, other_poly):
        if not isinstance(other_poly, GF2mPolynomial) or self.gf is not other_poly.gf:
            raise TypeError("Polynomials must be over the same GF(2^m) field.")
        
        max_len = max(len(self.coeffs), len(other_poly.coeffs))
        res_coeffs = [0] * max_len
        
        for i in range(len(self.coeffs)):
            res_coeffs[i] = self.gf.add(res_coeffs[i], self.coeffs[i])
        for i in range(len(other_poly.coeffs)):
            res_coeffs[i] = self.gf.add(res_coeffs[i], other_poly.coeffs[i])
            
        return GF2mPolynomial(self.gf, res_coeffs)

    def __mul__(self, other_poly):
        if not isinstance(other_poly, GF2mPolynomial) or self.gf is not other_poly.gf:
            raise TypeError("Polynomials must be over the same GF(2^m) field.")
        
        if self.is_zero() or other_poly.is_zero():
            return GF2mPolynomial(self.gf, [0]) # Product with zero is zero
        
        res_degree = self.degree() + other_poly.degree()
        res_coeffs = [0] * (res_degree + 1)
        
        for i, c1 in enumerate(self.coeffs):
            if c1 == 0: continue # Optimization
            for j, c2 in enumerate(other_poly.coeffs):
                if c2 == 0: continue # Optimization
                prod_term = self.gf.multiply(c1, c2)
                res_coeffs[i+j] = self.gf.add(res_coeffs[i+j], prod_term)
                
        return GF2mPolynomial(self.gf, res_coeffs)

    def eval_at(self, x_gf2m):
        """Evaluates the polynomial at a point x_gf2m in the field using Horner's method."""
        res = 0 # Initialize result (GF element)
        # Horner's: c_n*x^n + ... + c_1*x + c_0 = ((...(c_n*x + c_{n-1})*x + ...)*x + c_0)
        for coeff in reversed(self.coeffs): # Iterate from highest degree coeff (c_n) down to c_0
            res = self.gf.add(self.gf.multiply(res, x_gf2m), coeff)
        return res

    def __divmod__(self, divisor_poly):
        if not isinstance(divisor_poly, GF2mPolynomial) or self.gf is not divisor_poly.gf:
            raise TypeError("Polynomials must be over the same GF(2^m) field.")
        if divisor_poly.is_zero():
            raise ZeroDivisionError("Polynomial division by zero.")

        quotient = GF2mPolynomial(self.gf, [0])
        # Remainder starts as the dividend (self)
        remainder = GF2mPolynomial(self.gf, list(self.coeffs)) # Use a copy

        divisor_deg = divisor_poly.degree()
        divisor_leading_coeff = divisor_poly.coeffs[-1] # Highest degree coefficient of divisor
        
        # This check should ideally not be needed if _remove_leading_zeros works and divisor is not zero
        if divisor_leading_coeff == 0 : 
            raise ValueError("Divisor polynomial has leading zero coefficient but is not zero polynomial. This indicates an internal error.")

        divisor_leading_coeff_inv = self.gf.inverse(divisor_leading_coeff)

        # Long division loop
        while remainder.degree() >= divisor_deg and not remainder.is_zero():
            deg_diff = remainder.degree() - divisor_deg
            # Scale factor for the current term of the quotient
            scale = self.gf.multiply(remainder.coeffs[-1], divisor_leading_coeff_inv)
            
            # Construct current quotient term: scale * x^deg_diff
            current_q_term_coeffs = [0] * (deg_diff + 1)
            current_q_term_coeffs[deg_diff] = scale
            current_q_term_poly = GF2mPolynomial(self.gf, current_q_term_coeffs)
            quotient = quotient + current_q_term_poly # Add to overall quotient
            
            # Subtract (scale * x^deg_diff * divisor_poly) from remainder
            term_to_subtract_poly = current_q_term_poly * divisor_poly
            remainder = remainder + term_to_subtract_poly # In GF(2^m), add is subtract
            # remainder._remove_leading_zeros() # Important: ensure degree is updated correctly
                                            # This is handled by GF2mPolynomial constructor and ops
                                            
        return quotient, remainder
        
    def __str__(self):
        if self.is_zero(): return "0"
        terms = []
        # Iterate from highest degree down to constant term
        for i in range(len(self.coeffs) -1, -1, -1): 
            if self.coeffs[i] != 0:
                term = ""
                coeff_str = str(self.coeffs[i])
                # Coefficient part
                if self.coeffs[i] != 1 or i == 0 : # Show coeff if not 1 (unless it's x^0 term)
                    term += coeff_str
                elif i > 0 and self.coeffs[i] == 1: # Coeff is 1 and not x^0 term, omit '1'
                    pass 
                
                # Variable part
                if i > 0: term += "x" 
                if i > 1: term += f"^{i}" 
                terms.append(term)
        return " + ".join(terms) if terms else "0" # Should not be empty if not zero poly

    @classmethod
    def zero(cls, gf_field): return cls(gf_field, [0])


class ReedSolomonCodec_BW:
    """Reed-Solomon Encoder/Decoder using Berlekamp-Welch. Operates on binary lists."""

    _PRIMITIVE_POLYNOMIALS_FOR_M = {
        # m: polynomial (integer representation, e.g., x^3+x+1 is 0b1011)
        1: 0b11,       # x+1
        2: 0b111,      # x^2+x+1
        3: 0b1011,     # x^3+x+1
        4: 0b10011,    # x^4+x+1
        5: 0b100101,   # x^5+x^2+1
        6: 0b1000011,  # x^6+x+1
        7: 0b10001001, # x^7+x^3+1
        8: 0b100011101, # x^8+x^4+x^3+x^2+1 (standard AES/QR polynomial)
        # Using octal representations from a common source for larger m
        9: int("1021", 8),  # x^9 + x^4 + 1 (0o1021 = 0b1000010001)
        10: int("2011", 8), # x^10 + x^3 + 1 (0o2011 = 0b10000001001)
        11: int("4005", 8), # x^11 + x^2 + 1 (0o4005 = 0b100000000101)
        12: int("10123", 8),# x^12 + x^6 + x^4 + x + 1 (0o10123 = 0b1000001010011)
        13: int("20033", 8),# x^13 + x^4 + x^3 + x + 1 (0o20033 = 0b10000000011011)
        14: int("42103", 8),# x^14 + x^10 + x^6 + x + 1 (0o42103 = 0b100010001000011)
        15: int("100003", 8),#x^15 + x + 1 (0o100003 = 0b100000000000011)
        16: int("210013", 8) #x^16 + x^12 + x^3 + x + 1 (0o210013 = 0b10001000000010011)
    }

    def __init__(self, k_message_symbols: int, num_parity_symbols: int, m_bits_per_symbol: int):
        if k_message_symbols <= 0: raise ValueError("k_message_symbols must be positive.")
        if num_parity_symbols <= 0: raise ValueError("num_parity_symbols must be positive.")
        if m_bits_per_symbol <= 0: raise ValueError("m_bits_per_symbol must be positive.")

        self.k_sym = k_message_symbols
        self.num_parity_sym = num_parity_symbols
        self.n_sym = self.k_sym + self.num_parity_sym # Total symbols in codeword
        self.m = m_bits_per_symbol

        if self.m not in self._PRIMITIVE_POLYNOMIALS_FOR_M:
            raise ValueError(f"No predefined primitive polynomial for m={self.m}. Add one to _PRIMITIVE_POLYNOMIALS_FOR_M.")
        primitive_poly_int = self._PRIMITIVE_POLYNOMIALS_FOR_M[self.m]
        
        self.gf = GF2m(self.m, primitive_poly_int)
        # n_sym must be less than or equal to field order - 1 if using distinct non-zero field elements as eval points
        # Or less than field order if 0 can be an eval point (not typical for alpha^i form)
        if self.n_sym >= self.gf.order: # (2^m symbols available, 0 to 2^m-1)
                                       # If eval points are alpha^0 to alpha^(n-1), all non-zero.
                                       # Max n is order-1 for distinct non-zero points.
            raise ValueError(f"n_sym ({self.n_sym}) must be less than field order ({self.gf.order}) for m={self.m} to ensure distinct evaluation points from powers of alpha.")
        
        self.t_correctable = self.num_parity_sym // 2 # Max errors correctable
        if self.num_parity_sym % 2 != 0:
            # This is fine, num_parity_sym = 2t or 2t+1. B-W usually defined for 2t parity.
            # If num_parity_sym is odd, say 2t+1, we can still only correct t errors.
            # The extra parity symbol might help with error detection beyond t errors, but not correction.
            print(f"Warning: num_parity_symbols ({self.num_parity_sym}) is odd. Effective t_correctable is {self.t_correctable}.")

        # Choose a generator element for evaluation points, typically alpha = 2 (which is 'x')
        self.alpha = 2 
        if self.alpha == 0 or self.alpha >= self.gf.order: 
            # Check if alpha=2 is valid (e.g. if m=1, order=2, alpha=2 is not in field {0,1})
            # For m=1, alpha_to = [1], index_of = [-1, 0]. alpha=2 not in index_of.
            # A common choice is a primitive element of the field.
            # For GF(2^m), 'x' (represented as 2) is often primitive if poly is primitive.
            try:
                self.gf.index_of[self.alpha] # Check if alpha is a valid non-zero field element
            except IndexError:
                 raise ValueError(f"Generator element alpha={self.alpha} not valid for GF(2^{self.m}) field elements.")
            if self.gf.index_of[self.alpha] == -1 and self.alpha !=0 : # Should be caught by IndexError if out of bounds
                 raise ValueError(f"Generator element alpha={self.alpha} has no log, not suitable for GF(2^{self.m})")


        # Evaluation points: alpha^0, alpha^1, ..., alpha^(n-1)
        self.eval_points = [self.gf.power(self.alpha, i) for i in range(self.n_sym)]
        # Ensure all eval points are distinct (guaranteed if n_sym <= order-1 and alpha is primitive)
        if len(set(self.eval_points)) != self.n_sym:
            raise ValueError("Evaluation points are not distinct. Check alpha or n_sym relative to field order.")


    def _bits_to_symbols(self, bit_list: list) -> list:
        """Converts a list of 0s/1s into a list of m-bit symbols (integers)."""
        if len(bit_list) % self.m != 0:
            raise ValueError(f"Bit list length {len(bit_list)} must be a multiple of m={self.m}.")
        if not all(bit in (0, 1) for bit in bit_list):
            raise ValueError("Input list must contain only 0s or 1s.")
            
        num_symbols = len(bit_list) // self.m
        symbols = []
        for i in range(num_symbols):
            symbol_val = 0
            for j in range(self.m):
                symbol_val = (symbol_val << 1) | bit_list[i*self.m + j]
            symbols.append(symbol_val)
        return symbols

    def _symbols_to_bits(self, symbols: list) -> list:
        """Converts a list of symbols (integers) into a list of 0s/1s."""
        bit_list = []
        for sym in symbols:
            if not (0 <= sym < self.gf.order): # Check symbol validity
                raise ValueError(f"Symbol {sym} out of range for GF(2^{self.m})")
            # Format symbol to binary string of length m, padding with leading zeros
            bin_str = format(sym, f'0{self.m}b')
            bit_list.extend([int(b) for b in bin_str])
        return bit_list

    # ===== Addition: Polynomial scaled by a scalar =====
    def _scale_polynomial(self, poly: "GF2mPolynomial", scalar: int) -> "GF2mPolynomial":
        """scalar * poly(coefficient-wise multiplication over GF)"""
        if scalar == 0 or poly.is_zero():
            return GF2mPolynomial.zero(self.gf)
        scaled = [self.gf.multiply(c, scalar) for c in poly.coeffs]
        return GF2mPolynomial(self.gf, scaled)

    # ===== Lagrange interpolation (k points ⇒ unique polynomial of degree < k) =====
    def _lagrange_interpolate(
        self, xs: List[int], ys: List[int]
    ) -> "GF2mPolynomial":
        """
        Return P(x), which satisfied P(xs[i]) == ys[i], deg(P) < len(xs)
        Naive O(k^2) implementation; sufficient for k ≤ 12(or customizable).
        """
        k = len(xs)
        assert k == len(ys)
        gf = self.gf

        P = GF2mPolynomial.zero(gf)  # Result of add up

        for i in range(k):
            # Construct the basis polynomial l_i(x)
            numer = GF2mPolynomial(gf, [1])   # Product (x - x_j)
            denom = 1                         # Product  (x_i - x_j)

            for j in range(k):
                if i == j:
                    continue
                # (x - x_j)  => coefficients [x_j, 1]  (since char=2, subtraction equals addition)
                factor = GF2mPolynomial(gf, [xs[j], 1])
                numer = numer * factor
                denom = gf.multiply(
                    denom, gf.add(xs[i], xs[j])
                )  # (x_i - x_j) = x_i + x_j

            denom_inv = gf.inverse(denom)
            l_i = self._scale_polynomial(numer, denom_inv)  # full l_i(x)

            term = self._scale_polynomial(l_i, ys[i])       # y_i * l_i(x)
            P = P + term

        return P  # deg(P) < k

    # =================  Encode =================
    def encode(self, message_bit_list: List[int]) -> List[int]:
        """
        Systematic encoding: codeword symbols satisfy The remaining symbols 
        are obtained via interpolation and still comply with the  RS code constraint.
        """
        expected_len = self.k_sym * self.m
        if len(message_bit_list) != expected_len:
            raise ValueError(
                f"Message bit list length must be {expected_len} "
                f"(k={self.k_sym} * m={self.m}). Got {len(message_bit_list)}."
            )

        # Split the message into symbols of m bits each
        message_symbols = self._bits_to_symbols(message_bit_list)

        # 1. Take the first k evaluation points and message values for Lagrange interpolation
        xs = self.eval_points[: self.k_sym]
        ys = message_symbols
        P_poly = self._lagrange_interpolate(xs, ys)  # deg < k

        # 2. Evaluate the interpolated polynomial at all n evaluation points → codeword symbols
        codeword_symbols = [P_poly.eval_at(pt) for pt in self.eval_points]

        # Verify systematic property (optional and removable)
        assert codeword_symbols[: self.k_sym] == message_symbols

        # 3. Symbols to bits
        return self._symbols_to_bits(codeword_symbols)


    def _solve_berlekamp_welch_system(self, Y_received_symbols: list, t_errors_to_correct: int):
        """
        Sets up and solves the Berlekamp-Welch system of equations:
        N(X_i) - Y_i * Q(X_i) = 0
        where Q(x) is monic of degree t: Q(x) = x^t + q_{t-1}x^{t-1} + ... + q_0
        N(x) has degree k+t-1: N(x) = N_{k+t-1}x^{k+t-1} + ... + N_0
        The equations become:
        sum_{j=0}^{k+t-1} N_j X_i^j - Y_i * (sum_{j=0}^{t-1} q_j X_i^j) = Y_i * X_i^t
        Unknowns: q_0, ..., q_{t-1} (t variables)
                  N_0, ..., N_{k+t-1} (k+t variables)
        Total variables: t + (k+t) = k+2t
        Number of equations: n_sym (which should be k+2t for a determined system)
        """
        n = self.n_sym # Number of evaluation points / equations
        k = self.k_sym # Number of message symbols
        t = t_errors_to_correct

        # Number of unknown coefficients for Q(x) (excluding the monic x^t term)
        num_q_vars = t 
        # Number of unknown coefficients for N(x)
        num_n_vars = k + t 
        total_vars = num_q_vars + num_n_vars # These are the columns in matrix A

        if n < total_vars:
            # This implies n_sym < k+2t, system is underdetermined.
            # Standard RS codes often have n_sym = k+2t. If n_sym > k+2t, it's overdetermined.
            # If n_sym (num_parity_sym + k_sym) is not equal to k+2t, there's a mismatch.
            # self.num_parity_sym = 2t. So n_sym = k_sym + num_parity_sym = k + 2t.
            # This condition should not be met if parameters are consistent.
            raise UncorrectableError_RS(f"System underdetermined: n_sym ({n}) < total_vars ({total_vars}). Check RS parameters.")

        matrix_A = [] # Stores coefficients of unknowns
        vector_b = [] # Stores the RHS of equations (Y_i * X_i^t)

        # Constructing the matrix for [q_0, ..., q_{t-1}, N_0, ..., N_{k+t-1}]
        # Equation i:  -Y_i * sum_{j=0}^{t-1} q_j X_i^j  + sum_{l=0}^{k+t-1} N_l X_i^l = Y_i * X_i^t
        for i in range(n): # For each evaluation point X_i and received symbol Y_i
            X_i = self.eval_points[i]
            Y_i = Y_received_symbols[i]
            
            row = [0] * total_vars
            
            # Coefficients for q_j terms: -Y_i * X_i^j (or Y_i*X_i^j in GF2m)
            current_Xi_power_for_q = 1 # For X_i^0
            for j_q in range(num_q_vars): # j_q from 0 to t-1 (for q_0 to q_{t-1})
                # Term is Y_i * X_i^j for q_j (sign is + in GF2m)
                row[j_q] = self.gf.multiply(Y_i, current_Xi_power_for_q)
                current_Xi_power_for_q = self.gf.multiply(current_Xi_power_for_q, X_i)
            
            # Coefficients for N_l terms: X_i^l
            current_Xi_power_for_n = 1 # For X_i^0
            for j_n in range(num_n_vars): # j_n from 0 to k+t-1 (for N_0 to N_{k+t-1})
                # Term is X_i^l for N_l (but we need to be careful with signs if not GF2m)
                # The equation is N(X_i) - Y_i Q(X_i) = 0
                # N_coeffs are positive, Q_coeffs are negative (or positive in GF2m)
                # My system was: sum(N_l X_i^l) + sum((Y_i X_i^j) q_j) = Y_i X_i^t
                # So N_l terms have X_i^l coeffs.
                # The previous code had N_l terms as positive, q_j terms as positive.
                # Let's stick to N(x) = Y(x)Q(x) => N(x) - Y(x)Q(x) = 0
                # N_coeffs are [N_0, ..., N_{k+t-1}]
                # Q_error_coeffs are [q_0, ..., q_{t-1}] for Q(x) = x^t + sum q_j x^j
                # sum_{l=0}^{k+t-1} N_l X_i^l - Y_i * (X_i^t + sum_{j=0}^{t-1} q_j X_i^j) = 0
                # sum_{l=0}^{k+t-1} N_l X_i^l - sum_{j=0}^{t-1} (Y_i X_i^j) q_j = Y_i X_i^t
                # Variables: N_0 .. N_{k+t-1}, q_0 .. q_{t-1}
                # My previous code:
                # row[j_q] for Q terms, row[num_q_vars + j_n] for N terms
                # This means solution_vars = [q_coeffs_partial, N_coeffs]
                # The system solved was:
                # sum_{idx_q=0}^{t-1} Coeff_Q[idx_q]*q[idx_q] + sum_{idx_n=0}^{k+t-1} Coeff_N[idx_n]*N[idx_n] = RHS
                # Coeff_Q[j_q] = Y_i * X_i^{j_q}
                # Coeff_N[j_n] = X_i^{j_n}
                # This matches: sum (Y_i X_i^j)q_j + sum (X_i^l)N_l = Y_i X_i^t
                # This is correct for GF(2^m) where -A = A.
                row[num_q_vars + j_n] = current_Xi_power_for_n 
                current_Xi_power_for_n = self.gf.multiply(current_Xi_power_for_n, X_i)

            matrix_A.append(row)
            
            # RHS: Y_i * X_i^t
            b_term = self.gf.multiply(Y_i, self.gf.power(X_i, t))
            vector_b.append(b_term)
            
        try:
            # Solve Ax = b for x = [q_0,...,q_{t-1}, N_0,...,N_{k+t-1}]
            solution_vars = self._gaussian_elimination_gf2m(matrix_A, vector_b)
        except UncorrectableError_RS as e: # Catch specific error from Gaussian elimination
            raise UncorrectableError_RS(f"Gaussian elimination failed for B-W system: {e}", Y_received_symbols)
        except Exception as e: # Catch any other unexpected error
            raise UncorrectableError_RS(f"Unexpected error during Gaussian elimination for B-W system: {e}", Y_received_symbols)

        if solution_vars is None: # Should be caught by UncorrectableError_RS from Gaussian elim if inconsistent
            raise UncorrectableError_RS("B-W system unsolvable or has no unique solution (Gaussian elimination returned None).", Y_received_symbols)

        # Extract coefficients for Q(x) and N(x) from solution_vars
        # q_coeffs_partial are [q_0, q_1, ..., q_{t-1}]
        q_coeffs_partial = solution_vars[:num_q_vars]    
        # N_coeffs are [N_0, N_1, ..., N_{k+t-1}]
        N_coeffs         = solution_vars[num_q_vars:] 
        
        # Construct full Q(x) coefficients: [q_0, ..., q_{t-1}, 1] (for x^t)
        Q_coeffs = q_coeffs_partial + [1] # Q(x) is monic: q_0 + q_1*x + ... + q_{t-1}*x^{t-1} + 1*x^t
        
        return Q_coeffs, N_coeffs

    def _gaussian_elimination_gf2m(self, matrix_A_orig, vector_b_orig):
        """
        Solves a system of linear equations Ax = b over GF(2^m) using Gaussian elimination.
        matrix_A: list of lists (rows)
        vector_b: list (RHS)
        Returns solution list x, or raises UncorrectableError_RS if no unique solution/inconsistent.
        """
        matrix_A = [list(row) for row in matrix_A_orig] # Make a mutable copy
        vector_b = list(vector_b_orig) # Make a mutable copy

        num_equations = len(matrix_A)
        if num_equations == 0: return []
        num_variables = len(matrix_A[0])

        if len(vector_b) != num_equations:
            raise ValueError("Matrix A and vector b dimensions mismatch.")

        pivot_row = 0
        # For back substitution, keep track of which column corresponds to pivot in each row
        pivot_cols = [-1] * num_equations 

        # Forward elimination
        for col in range(num_variables):
            if pivot_row >= num_equations: break # All rows processed or pivoted

            # Find a row with a non-zero entry in the current column (col) at or below pivot_row
            i = pivot_row
            while i < num_equations and matrix_A[i][col] == 0:
                i += 1

            if i < num_equations: # Found a suitable pivot
                # Swap row i with pivot_row
                matrix_A[pivot_row], matrix_A[i] = matrix_A[i], matrix_A[pivot_row]
                vector_b[pivot_row], vector_b[i] = vector_b[i], vector_b[pivot_row]

                pivot_val = matrix_A[pivot_row][col]
                # pivot_val should not be zero here due to the search loop
                if pivot_val == 0: # Should ideally not happen
                    raise UncorrectableError_RS("Internal error: Pivot element became zero unexpectedly during Gaussian elimination.")
                
                pivot_inv = self.gf.inverse(pivot_val)
                pivot_cols[pivot_row] = col # Record pivot column for this row

                # Normalize pivot row (make pivot element 1)
                for j in range(col, num_variables): # Start from pivot column
                    matrix_A[pivot_row][j] = self.gf.multiply(matrix_A[pivot_row][j], pivot_inv)
                vector_b[pivot_row] = self.gf.multiply(vector_b[pivot_row], pivot_inv)

                # Eliminate other non-zero entries in the current column (col)
                for i_row_below in range(num_equations):
                    if i_row_below != pivot_row:
                        factor = matrix_A[i_row_below][col]
                        if factor != 0: # If there's something to eliminate
                            # Subtract factor * pivot_row from i_row_below
                            # (or add, since it's GF2m)
                            for j_col in range(col, num_variables): # Start from pivot column
                                term = self.gf.multiply(factor, matrix_A[pivot_row][j_col])
                                matrix_A[i_row_below][j_col] = self.gf.add(matrix_A[i_row_below][j_col], term)
                            term_b = self.gf.multiply(factor, vector_b[pivot_row])
                            vector_b[i_row_below] = self.gf.add(vector_b[i_row_below], term_b)
                pivot_row += 1
        
        # Check for inconsistency (e.g., 0x_1 + 0x_2 = C where C != 0)
        for i_row in range(pivot_row, num_equations): # Check rows that should be all zero in A
            if vector_b[i_row] != 0:
                # This means we have an equation like 0 = non-zero
                raise UncorrectableError_RS("System inconsistent (0 = non-zero after forward elimination).")

        # If pivot_row < num_variables, system has free variables (infinitely many solutions for Ax=b)
        # For Berlekamp-Welch, we expect n_sym equations and k+2t variables.
        # If n_sym = k+2t, we expect a unique solution if rank is full.
        # If rank < num_variables (i.e. pivot_row < num_variables), it implies non-unique solution.
        # The B-W algorithm might sometimes yield non-unique solutions if e < t, but one should work.
        # However, this Gaussian elimination aims for a unique one if possible.
        # If pivot_row < num_variables, it means there are free variables.
        # The current B-W setup expects a unique solution.
        if pivot_row < num_variables and num_equations >= num_variables : # More vars than pivots, but enough equations for unique
             # This might indicate issues if a unique solution was expected.
             # For now, proceed to back-substitution; free variables will be 0 if not pivoted.
             pass


        # Back substitution
        solution = [0] * num_variables
        for i in range(pivot_row - 1, -1, -1): # Iterate from last pivot row upwards
            p_col = pivot_cols[i]
            if p_col == -1: continue # Should not happen if pivot_row was incremented correctly

            val = vector_b[i] # Start with RHS value for this row
            # Subtract sum of (A[i][j] * solution[j]) for j > p_col
            for j in range(p_col + 1, num_variables):
                term = self.gf.multiply(matrix_A[i][j], solution[j])
                val = self.gf.add(val, term) # Add is subtract in GF(2^m)
            solution[p_col] = val
            
        return solution


    def decode(
        self, received_bit_list: List[int]
    ) -> Tuple[List[int], List[int], int]:
        """
        Returns:

        corrected_message_bit_list (same length and order as the original message)
        corrected_codeword_bit_list (full length 𝑛×𝑚, in bits) actual_symbol_errors_found (number of corrected symbols)
        """
        expected_len = self.n_sym * self.m
        if len(received_bit_list) != expected_len:
            raise ValueError(
                f"Received bit list length must be {expected_len} "
                f"(n={self.n_sym} * m={self.m}). Got {len(received_bit_list)}."
            )

        received_symbols = self._bits_to_symbols(received_bit_list)

        t = self.t_correctable  # correctable symbols

        # --------- Simplified case when t==0 (preserve original behavior) ----------
        if t == 0:
            message_candidate_bits = received_bit_list[: self.k_sym * self.m]
            reenc = self.encode(message_candidate_bits)
            if reenc == received_bit_list:
                return (
                    message_candidate_bits,
                    received_bit_list,
                    0,
                )
            raise UncorrectableError_RS(
                "Errors present but t=0 (no correction possible).",
                received_bit_list,
            )

        # --------- B-W Decoding ----------
        try:
            Q_coeffs_bw, N_coeffs_bw = self._solve_berlekamp_welch_system(
                received_symbols, t
            )
        except UncorrectableError_RS as e:
            e.received_input = received_bit_list
            raise
        Q_poly = GF2mPolynomial(self.gf, Q_coeffs_bw)
        N_poly = GF2mPolynomial(self.gf, N_coeffs_bw)

        try:
            msg_poly, rem_poly = divmod(N_poly, Q_poly)
        except ZeroDivisionError:
            raise UncorrectableError_RS(
                "Q(x) is zero during division.", received_bit_list
            )
        if not rem_poly.is_zero():
            raise UncorrectableError_RS(
                "N(x) mod Q(x) ≠ 0 ⇒ more than t errors或 B-W failed.",
                received_bit_list,
            )

        # ---------- **Systematic code only: extract message from evaluations instead of coefficients** ----------
        message_symbols_corrected = [
            msg_poly.eval_at(pt) for pt in self.eval_points[: self.k_sym]
        ]
        corrected_message_bit_list = self._symbols_to_bits(
            message_symbols_corrected
        )

        # Reconstruct full codeword (evaluation of the polynomial)
        corrected_codeword_symbols = [
            msg_poly.eval_at(pt) for pt in self.eval_points
        ]
        corrected_codeword_bit_list = self._symbols_to_bits(
            corrected_codeword_symbols
        )

        # Count the number of corrected errors
        actual_symbol_errors_found = sum(
            1
            for r, c in zip(received_symbols, corrected_codeword_symbols)
            if r != c
        )
        return (
            corrected_message_bit_list,
            corrected_codeword_bit_list,
            actual_symbol_errors_found,
        )
