# DeepLearning-GameBot

DeepLearning Game Bot is a competition winning game bot, it consists of a deep learning based computer vision system and a controller. The bot is adoptable and training via Google Collab, developed completely remotely. The computer vision model and controller for this agent has won 5th place in a competition. It's a simple model inspired by the Yolo framework.

Time spent: **12** hours spent in total

## User Stories

The following **required** functionality is completed:

* [x] Develop a controller that interface with game broadcast **nodes and topics**
* [x] Generate data set based on the **developed controller**
* [x] Play **against the default ai** from the tournament
* [x] Controller **refined** to include special counter measures.

The following **optional** features are implemented:

* [x] Win competitions


## Video Walkthrough

Here's a walkthrough of training the computer vision model using racing (TOP 5):

<img src='DLCompetition.gif' title='Vision model development' width='' alt='Refining the initial vision model' />

Here's a walkthrough of the final model using a controller to play hockey:

<img src='DLTournament.gif' title='Controller model development' width='' alt='Competing with other agents' />

Scaffolding code derived by the work of Prof. Philipp Krähenbühl. Thank you for the inspiration and the teachings!

## Notes

Colab has a limitation in terms of X11 forwarding which made it difficult to stream the activities in real time. The overall hardware was limited for training, using better hardware could yield in better results with more comprehensive models. The best approach is to generate an MP4 file and download it from the remote server. Instead of using a controller, with better hardware, could have implemented a reinforcement learning approach.

## License

    Copyright 2021

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific languag
