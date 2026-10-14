"""Evaluate one saved DQN model ten times with the official CartPole test.

Change trained_nn below, then run: python evaluate.py
"""

import csv
import os

import gym
import numpy as np
import pandas as pd
import torch

from test_script import QNetwork, test_pole_length

# copy of test_script(), but runs 10 times instead
def main():
    trained_nn = 'semih_vanilla_dqn.pth'

    evaluations = 10
    trials_per_length = 10
    pole_lengths = np.linspace(0.4, 1.8, 30)

    model = QNetwork(4, 2)
    model.load_state_dict(torch.load(
        'weights/' + trained_nn, map_location='cpu', weights_only=True
    ))
    model.eval()

    model_name = os.path.splitext(trained_nn)[0]
    results_file = f'results/data/{model_name}_experiment_results.xlsx'
    pole_file = f'results/data/{model_name}_pole_results.csv'
    trials_file = f'results/data/{model_name}_trial_results.csv'

    experiment_results = []

    with open(pole_file, 'w', newline='', encoding='utf-8') as pole_csv, \
         open(trials_file, 'w', newline='', encoding='utf-8') as trials_csv:

        pole_writer = csv.writer(pole_csv)
        trial_writer = csv.writer(trials_csv)
        pole_writer.writerow(['evaluation', 'pole_length', 'average', 'std'])
        trial_writer.writerow(['evaluation', 'pole_length', 'trial', 'score'])

        for evaluation in range(1, evaluations + 1):
            results = {'Evaluation': evaluation}
            total_score = 0

            print(f'Evaluation {evaluation}/{evaluations}', flush=True)

            for length in pole_lengths:
                scores = []

                for trial in range(1, trials_per_length + 1):
                    env = gym.make('CartPole-v1')
                    try:
                        env.unwrapped.length = float(length)
                        # Use the original evaluation function without changing it
                        with torch.no_grad():
                            score = test_pole_length(env, model)
                    finally:
                        env.close()

                    scores.append(score)
                    trial_writer.writerow([evaluation, float(length), trial, score])

                average = float(np.mean(scores))
                std = float(np.std(scores))
                pole_writer.writerow([evaluation, float(length), average, std])
                trials_csv.flush()
                pole_csv.flush()

                results[f'Avg_{round(float(length), 2)}'] = average
                results[f'Std_{round(float(length), 2)}'] = std
                total_score += average

            results['Total'] = total_score
            results['Overall'] = total_score / len(pole_lengths)
            experiment_results.append(results)

            pd.DataFrame(experiment_results).to_excel(results_file, index=False)

if __name__ == '__main__':
    main()
