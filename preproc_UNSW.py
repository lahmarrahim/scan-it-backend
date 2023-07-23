import numpy as np
def apply_log_transformation(dataset):
    columns=["spkts", "dpkts","sbytes","dbytes","rate","sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit","stcpb","dtcpb","smean","dmean","response_body_len"]
    eps = 1e-5
    for column in columns:
        dataset[column] = np.log(dataset[column] + eps)
    return dataset
