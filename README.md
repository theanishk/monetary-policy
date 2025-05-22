# monetary-policy
Authors: [Anish Kumar, 21322006](a_kumar2@hs.iitr.ac.in), [Arunesh Pratap Singh Tomar, 21322009](a_pstomar@hs.iitr.ac.in)

This project is done for the course HSN-620 - Analytics for Business and Society at Indian Institute of Technology, Roorkee.

Objective - This project aims to get the optimal monetary policy for US economy during crisis.

To run the code you need to install the libraries mentioned in the requirements.txt file. You can do this by running the following command:
```bash
pip install -r requirements.txt
```

## Data
The time period of the data is from 1990 to 2024 for US economy. 
The data is collected from FRED (Federal Reserve Economic Data).
The data is in the `data` folder. The data is in Excel format. The data is cleaned and preprocessed in the `data_preprocessing.py` file. The cleaned data is saved as `data/us_macro_data.xlsx`.

## Code
There exist a notebook for Exploratory Data Analysis in `notebooks/EDA.ipynb`. The notebook is used to visualize the data and understand the data better. 

The codes are in the `scripts` folder. The code is divided into different files for different tasks. The main file is `train.py`. The other files are:
- `data_preprocessing.py`: This file is used to preprocess the data. It loads the data from the `data` folder, cleans it, and saves it as `data/us_macro_data.xlsx`.
- `taylor.py`: This file is used to calculate the Taylor rule for the US economy. It takes the cleaned data as input and saves the output as `data/taylor_rule_result.xlsx`.
- `environment_ann.py`: This file is used to create the environment for the reinforcement learning model.
- `actor_critic.py`: This file is used to create the actor-critic model for the reinforcement learning model using ANN.
- `ddpg_agent.py`: This file is used to implement the DDPG algorithm for the reinforcement learning model.
- `train.py`: This file is used to train the reinforcement learning model. It takes the environment and the actor-critic model as input and trains the model.

## Figures
The figures are saved in the `figures` folder. These figures are generated while using the above scripts and notebooks.

## Runs
The runs are saved in the `runs` folder. The runs are saved as images and best models.

## Evaluation and Results
The evaluation for the Reinforcement Learning model is saved in the `evaluation_results` folder. The evaluation is done data.

For the results, the reinforcement learning model compared to the Taylor rule performed better by 0.7% in terms of MSE. But overall during crisis period the RL model overshot the interest rate above 100% which is not acceptable. The Taylor rule performed better in terms of stability and less overshooting.