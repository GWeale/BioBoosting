# BioBoost: A Custom Boosting Tree Implementation for Bioinformatics

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Original Implementation](#original-implementation)
  - [Extended Functionality](#extended-functionality)
    - [Hyperparameter Tuning and Cross-Validation](#hyperparameter-tuning-and-cross-validation)
    - [Making Predictions with a Saved Model](#making-predictions-with-a-saved-model)
- [Data](#data)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

**BioBoost** is a custom implementation of a boosting tree algorithm tailored for bioinformatics data, such as gene expression datasets. It is designed to be lightweight, efficient, and easily extensible without relying on specialized machine learning libraries.

## Features

- **Custom Boosting Implementation:** Build and train boosting trees from scratch.
- **Cross-Validation:** Evaluate model performance using k-fold cross-validation.
- **Hyperparameter Tuning:** Optimize model parameters to achieve the best performance.
- **Model Persistence:** Save and load trained models for future use.
- **Feature Engineering:** Automatic creation of polynomial interaction features.
- **Visualization:** Plot training loss and feature importances.
- **Lightweight Dependencies:** Utilizes basic Python packages for ease of use.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/CowCurry/BioBoost.git
   cd BioBoost
