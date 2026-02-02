The zip file includes the code and interactive dashboard that were created to visualize player performance ratings, compare lineup scenarios and generate optimal recommendations.
The **rugby_analysis.ipynb file** contains the source code that performs the data preprocessing, model training (Ridge vs Lasso comparison)
The **rugby_streamlit_dashboard.py file** contains the interactive Streamlit application that allows coaches to:
                    - View all player ratings and their classification points
                    - Input opponent strength (or use average opponent)
                    - Filter by available players (e.g., exclude injured players)
                    - Generate optimal lineup recommendations ranked by predicted goals per minute
                    - Visualize the lineups under the 8.0 point constraint
