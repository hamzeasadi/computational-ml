import numpy as np



# calculate P(A|B) given P(A), P(B|A), P(B|not A)
def P_A_given_B(p_A, p_B_A, p_B_notA):
    p_notA = 1 - p_A
    p_a_b = (p_B_A*p_A)/(p_B_A*p_A + p_notA*p_B_notA)
    return p_a_b

def main():
    p_base = 0.002
    p_b_a = 0.88
    p_b_notA = 0.05

    p_a_b = P_A_given_B(p_A=p_base, p_B_A=p_b_a, p_B_notA=p_b_notA)
    print(f'P(A|B) = {p_a_b}')

if __name__ == '__main__':
    main()
