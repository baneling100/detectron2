import pickle
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

def main(index):
    payloads = xm.rendezvous("main", payload=pickle.dumps(index))
    payloads = [pickle.loads(payload) for payload in payloads]
    print(payloads)

if __name__ == "__main__":
    xmp.spawn(main, nprocs=8)