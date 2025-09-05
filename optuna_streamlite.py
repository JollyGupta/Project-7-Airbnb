import streamlit as st
import optuna
import optuna.visualization as vis
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout='wide')
st.title(" AutoML Dashboard with Optuna + Streamlit")

# === Load your data ===
uploaded_file = st.file_uploader("Upload a cleaned CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully.")
    st.write("Shape:", df.shape)

    # === Target Selection ===
    target_col = st.selectbox("Select target column-Price:", df.columns)
    X = df.drop(columns=target_col)
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get column types
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # ========== OPTUNA OBJECTIVE FUNCTION ==========
    def objective(trial):
        model_name = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'CatBoost'])

        if model_name == 'RandomForest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            }
            model = RandomForestRegressor(**params, random_state=42)

        elif model_name == 'XGBoost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
            model = XGBRegressor(**params, random_state=42, verbosity=0)

        else:
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            }
            model = CatBoostRegressor(**params, verbose=0, random_state=42)

        preprocessor = ColumnTransformer([
         ('num', num_pipeline, num_cols),
         ('bin', binary_pipeline, binary_cols),
         ('onehot', onehot_pipeline, onehot_cols),
         #('ord', ordinal_pipeline, ordinal_cols) 
         ('target',target__encoding_pipeline,high_card_cols)
      ])

        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        score = cross_val_score(pipe, X_train, y_train, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
        return -1 * score.mean()

    # ========== Run Optuna ==========
    n_trials = st.slider("Number of Optuna Trials", 5, 100, 30)

    if st.button("🚀 Run Optuna Tuning"):
        with st.spinner("Tuning in progress..."):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            st.success("Tuning complete!")

            st.write("📌 Best Model:", study.best_trial.params['model'])
            st.write("📌 Best Hyperparameters:", study.best_trial.params)

            # Visualizations
            st.subheader("📊 Optuna Visualizations")

            st.plotly_chart(vis.plot_optimization_history(study), use_container_width=True)
            st.plotly_chart(vis.plot_param_importances(study), use_container_width=True)
            st.plotly_chart(vis.plot_parallel_coordinate(study), use_container_width=True)
            st.plotly_chart(vis.plot_slice(study), use_container_width=True)

            # Optional: show contour only for some parameters
            selected_params = [k for k in study.best_trial.params if k != 'model']
            if len(selected_params) >= 2:
                st.plotly_chart(vis.plot_contour(study, params=selected_params[:2]), use_container_width=True)

            # Final model evaluation
            model_name = study.best_trial.params['model']
            best_params = study.best_trial.params.copy()
            best_params.pop('model')

            if model_name == 'RandomForest':
                model = RandomForestRegressor(**best_params)
            elif model_name == 'XGBoost':
                model = XGBRegressor(**best_params, verbosity=0)
            else:
                model = CatBoostRegressor(**best_params, verbose=0)

            preprocessor = ColumnTransformer([
                ('num', SimpleImputer(strategy='mean'), num_cols),
                ('cat', Pipeline([
                    ('imp', SimpleImputer(strategy='most_frequent')),
                    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_cols)
            ])

            final_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            final_pipeline.fit(X_train, y_train)
            y_pred = final_pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.subheader("Final Model Performance")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"R² Score: {r2:.2f}")

            # Plot Actual vs Predicted
            st.subheader("📉 Actual vs Predicted Plot")
            plot_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            fig, ax = plt.subplots()
            sns.scatterplot(x='Actual', y='Predicted', data=plot_df, ax=ax, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            st.pyplot(fig)
