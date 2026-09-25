# Building HVAC Mini-Project

**Independent Research Project — Data-Driven Building Energy Prediction and HVAC Optimisation**

This small project demonstrates how mechanical/HVAC engineering can be combined with Python, machine learning and optimisation for building-energy applications.

## Project question
Can a simple building model, data-driven temperature prediction and constrained HVAC optimisation reduce energy use while maintaining indoor comfort?

## Four steps
1. **Building model:** a reduced-order 2R–2C thermal model.
2. **Prediction:** compare persistence, linear regression and Random Forest for next-hour indoor temperature.
3. **Optimisation:** test a small set of HVAC actions and choose the lowest-energy action that keeps the next temperature within 20–24 °C.
4. **Flexibility:** evaluate electricity cost under an illustrative time-of-use price profile.

## Important scope
The data are synthetic and generated from the reduced-order model. The project is a reproducible portfolio study, not a validated building-energy model. The electricity prices are illustrative, not real tariffs.

## Run
Create a virtual environment, install `requirements.txt`, then open the notebooks in order:

`01_building_model.ipynb` → `02_prediction.ipynb` → `03_optimization.ipynb` → `04_flexibility.ipynb`

## Repository structure
```
src/
  building.py       # building physics
  simulation.py     # weather + thermostat simulation
  prediction.py     # ML prediction
  optimization.py   # simple constrained HVAC optimisation
  flexibility.py    # TOU cost calculation
notebooks/
  01_building_model.ipynb
  02_prediction.ipynb
  03_optimization.ipynb
  04_flexibility.ipynb
```

## Why this project is relevant
The project combines thermal-system modelling, HVAC knowledge, data analysis, machine learning and optimisation. These are the skills I want to extend toward building energy systems and intelligent HVAC control.
