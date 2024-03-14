import torch
import pickle
import numpy as np

# tensor = np.load('./chembl_encoded_valid.npy')
# # tensor = torch.load('./compounds_data_valid.pt')
# print(type(tensor))
# print(tensor.shape)
# # tensor = tensor.detach().cpu().numpy()
# # print(type(tensor))
# tensor = tensor.astype(np.float32)
# with open('./chembl_encoded_valid.p', 'wb') as f:
#     pickle.dump(tensor, f)

# tensor = np.load('./chembl_value_valid.npy')
# # tensor = tensor.detach().cpu().numpy()
# print(type(tensor))
# print(tensor.shape)
# tensor = tensor.astype(np.float32)
# with open('chembl_value_valid.p', 'wb') as f:
#     pickle.dump(tensor, f)


with open("./chembl_encoded_valid.p", "rb") as f:
    data = pickle.load(f)
print(data.dtype)
with open("./chembl_value_valid.p", "rb") as f:
    value = pickle.load(f)
value = value[:,[3,4,5]]
print(value[0:5])
print(value.dtype)
# with open("./compounds_data_valid.p", "rb") as f:
#     data = pickle.load(f)
# print(data.dtype)
# with open("./compounds_value_valid.p", "rb") as f:
#     value = pickle.load(f)
# print(value.dtype)