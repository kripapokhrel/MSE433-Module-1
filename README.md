# MSE433 Module 1 - Reproduction Instructions

This guide explains how to install dependencies and reproduce the results from the Wheelchair Rugby Analysis project.

---
## Prerequisites
- **Python**: 3.8 or higher (3.9+ recommended)
- **Jupyter**: Required for running the notebook

---

## Required Data Files

Place these CSV files in the **project root** (same directory as the code):

| File | Description |
|------|-------------|
| `stint_data.csv` | Stint-level game data (teams, goals, players, minutes) |
| `player_data.csv` | Player classifications and ratings |

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the notebook
jupyter notebook "rugby_analysis .ipynb"

# 3b. Run the Streamlit dashboard
streamlit run rugby_streamlit_dashboard.py
```
---

## Step-by-Step Instructions

### 1. Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Reproduce Results

#### A. Jupyter Notebook (`rugby_analysis .ipynb`)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `rugby_analysis .ipynb`
3. Run all cells: **Cell → Run All**

Reproduces: data preprocessing, Ridge vs Lasso comparison, player ratings, lineup analysis.

#### B. Streamlit Dashboard (`rugby_streamlit_dashboard.py`)

1. From the project directory:
   ```bash
   streamlit run rugby_streamlit_dashboard.py
   ```
2. Open the URL in your browser (typically `http://localhost:8501`)
3. Use sidebar: **Overview & EDA**, **Player Ratings**, **Lineup Optimizer**, **Player Rankings**

---

## Expected Output / What You Should See

### Jupyter Notebook

- **Data load**: Console prints `Data Loaded Successfully.` plus shapes (e.g., Stints: ~7448 rows × 14 cols, Players: ~144 rows × 2 cols).
- **Data preview**: A dataframe of stint data (game_id, h_team, a_team, minutes, h_goals, a_goals, home1–4, away1–4).
- **Plots**: Histograms, team performance bar charts, and model comparison visualizations.
- **Player ratings**: Tables of top/bottom players by net rating.
- **Model comparison**: Ridge vs Lasso results and coefficients.

### Streamlit Dashboard

After opening `http://localhost:8501`:

- **Sidebar**: Navigation (Overview & EDA, Player Ratings, Lineup Optimizer, Player Rankings) and data summary (Total Stints, Total Players, Unique Teams).
- **Overview & EDA**: Stint duration histogram, team performance bar chart, Canada performance metrics.
- **Player Ratings**: Top 15 / Bottom 15 player tables and a horizontal bar chart of top 20 players by net rating.
- **Lineup Optimizer**: Configurable venue, injured players, and opponent; "Generate Optimal Lineups" returns top 10 lineups with class points and predicted goals/min, plus a bar chart.
- **Player Rankings**: Tables by role (Attacker, Flex, Defender) and a Net Rating vs Raw +/- scatter plot.

If CSV files are missing, you will see: *"Error: CSV files not found. Please ensure 'stint_data.csv' and 'player_data.csv' are in the same directory."*

---

## Dependencies Reference

| Package | Minimum Version |
|---------|-----------------|
| streamlit | 1.28.0 |
| pandas | 1.5.0 |
| numpy | 1.21.0 |
| matplotlib | 3.6.0 |
| seaborn | 0.12.0 |
| scikit-learn | 1.2.0 |
| jupyter | 1.0.0 |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "CSV files not found" | Ensure `stint_data.csv` and `player_data.csv` are in the project directory |
| Port in use | `streamlit run rugby_streamlit_dashboard.py --server.port 8502` |
| Plot not showing (Jupyter) | Re-run the cell with `%matplotlib inline` |
