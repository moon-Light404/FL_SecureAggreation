from phe import paillier

public_key, private_key = paillier.generate_paillier_keypair()

# secret_list = [3,141592653, 300, -4.6e-12]
# encrypted_list = [public_key.encrypt(x) for x in secret_list]
#
# print(encrypted_list)
#
# print([private_key.decrypt(x) for x in encrypted_list])


a = 3.14
b = 2.22
a_enc = public_key.encrypt(a)
b_enc = public_key.encrypt(b)
print(a_enc)
print(b_enc)

print(private_key.decrypt(a_enc + b_enc))