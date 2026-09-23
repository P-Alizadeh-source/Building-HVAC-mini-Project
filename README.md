# Building HVAC Mini-Project

**Data-Driven Prediction and Optimal Operation of a Building HVAC System**

This is a compact, reproducible mini-project prepared as part of application material for a PhD position related to building-energy modelling, data-driven prediction and intelligent HVAC operation.

## Research Question

Can a data-driven predictive model combined with constrained optimisation reduce HVAC energy use while maintaining indoor thermal comfort under changing outdoor conditions and internal loads?

## What this project contains

- Simple first-order building thermal model
- Rule-based (thermostat) baseline controller
- Random Forest model that predicts next-step indoor temperature
- Constrained optimisation controller (receding horizon)
- Quantitative comparison of energy use and comfort

## Project Structure

- `src/` → Python code (model, simulation, prediction, optimisation)
- `notebooks/` → Jupyter notebooks that show the results
- `requirements.txt` → list of Python packages

## How to run

1. Create a virtual environment
2. Activate it
3. Install the packages with: `pip install -r requirements.txt`
4. Open the notebooks in order (01 → 02 → 03)

## Key Results

- The prediction model has very low error (\~0.2 °C)
- The optimised controller keeps temperature inside the comfort band much better than the rule-based controller
- There is a clear energy vs comfort trade-off

## Notes

This is a simple portfolio mini-project made to understand and demonstrate the main ideas of the research topic.
