## Project Structure

The repository is organized as follows:

- `D3SAC.py`: Main training script for training and evaluating the D3SAC agent.
- `D3SAC_Gen.py`: Generalization testing script for evaluating the trained model's performance with varying numbers of users.
- `D3SAC_Real.py`: Real-world testing script that uses real-world datasets for model validation.
- `D3SAC_Gen_Real.py`: Real-world generalization testing script that evaluates the model's adaptability to different user scales using real-world datasets.
- `EdgeEnv.py`: Defines the MEC environment, including the state and action spaces, reward function, and simulation dynamics.
- `MD.py`: Implements the Mobile Device (MD) class, which manages task generation, data transmission, and local computation.
- `BS.py`: Implements the Base Station (BS) class, which represents the edge servers.
- `ReplayMemory.py`: Implements the replay memory for storing and sampling experiences for training the D3SAC agent.
- `para.py`: Contains all the simulation parameters, such as the number of MDs and BSs, learning rates, and weights for the reward function.
- `dataset/`: Contains the mobility data for the MDs. Each file in the `MDT` subdirectory corresponds to a single MD and contains its location over time. The data originates from https://dx.doi.org/10.21227/y2ry-rd58.
- `models/`: Directory for saving and loading trained models.
- `result/`: Directory for storing simulation results.

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.10.1 
- NumPy 1.19.2
- gym 0.19.0
- scipy==1.5.2

## Usage

To train the D3SAC agent, run the main script:

```bash
python D3SAC.py
```

The script will load the simulation parameters from `para.py`, create the MEC environment, and start the training process. The trained model will be saved in the `models/` directory, and the simulation results will be stored in the `result/` directory.

You can also run the other variants of the main script:

```bash
python D3SAC_Gen.py
python D3SAC_Real.py
python D3SAC_Gen_Real.py
```
