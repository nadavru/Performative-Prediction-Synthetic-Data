# Performative Prediction project - Synthetic Data

This repo is part of 236608 project at the Technion. 

## Installation

The following packages are required:

```bash
torch
numpy
matplotlib # for visualization
sklearn
pandas
```

## Usage

```bash
./run_script_tran1.sh i # i is the optimizer index - 1~7
```

The indexes refer to the following:

1 -> RRM\
2 -> RRM + ADAM (0.1)\
3 -> RRM + ADAM (0.1) + lookahead\
4 -> RRM + ADAM (0.1) + learned lookahead\
5 -> RRM + ADAM (0.9)\
6 -> RRM + ADAM (0.9) + lookahead\
7 -> RRM + ADAM (0.9) + learned lookahead

## License

[MIT](https://choosealicense.com/licenses/mit/)