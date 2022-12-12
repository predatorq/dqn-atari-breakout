# dqn_and_improve

This project is modified from https://gitee.com/goluke/dqn-breakout, and is a midterm assignment for Reinforcement Learning in SYSU.

PER DQN and Duel DQN are implemented on the basis of source code.

In this project, given a full implementation of a Nature-DQN agent, I boost the training speed using Prioritized Experience Replay and improve the performance using Dueling DQN

## Description

- ./duel-dqn is the model using Duel DQN.

- ./per-dqn is the model using PER DQN.

- ./per-duel-dqn is the model using Duel and PER.

- ./display.ipynb is the code to paint the pictures and show the results.

## Installation

- First fork the source code, and then clone to the local 

- Enter the local directory:   `cd dqn-breakout`

- Install the requirements: `pip install -r requirements.freezed`

- Due to authorization reasons, atari-py does not have a built-in Atari game ROM, it needs to be imported manually.

- Use the command below to import the roms:
  
  ```python
    python -m atari_py.import_roms ~/dqn-breakout/ROMS/
  ```

- Now the project can be run using: `sh run.sh`
