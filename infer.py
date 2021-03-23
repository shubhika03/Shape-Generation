import torch
import argparse

import numpy as np

import model
import dataset

def get_args():
    parser = argparse.ArgumentParser(description="Eval options")
    parser.add_argument("--model_path", type=str, default="./models/exp1/weights_50",
                        help="Path to save models")
    return parser.parse_args()

def make_some_noise(batch_size):
    return torch.rand(batch_size, 100)

def main():
    args = get_args()

    models = {}
    models['generator'] = model.Generator()
    models['generator'].load_state_dict(torch.load(args.model_path+'/generator.pth'))

    models['generator'].eval()

    store= []
    with torch.no_grad():

        for _ in range(100):

            random_input = make_some_noise(1)
            coeff = models['generator'](random_input)
            store.append(coeff.numpy()/10)

    store = np.array(store)
    np.save("generated_coeff.npy", store)

    

main()