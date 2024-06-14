from shiny import ui, render, App, reactive
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import mlflow
import pandas as pd
import os

# available selections

qs_strategies = ["random", "uncertainty_sampling", "probabilistic_active_learning"]
models = ["logistic_regression", "parzen_window_classifier", "random_forest_tree"]
datasets_image = ["cifar10"]
datasets_text = ["trec6"]
datasets_tabular = ["iris"]
datasets = datasets_image + datasets_text + datasets_tabular
batch_sizes = [1,10,25,50,100,1000]

# initialies selected list

selected_datasets_list = []
selected_models_list = []
selected_qs_list = []
selected_batch_sizes_list = []

selected_experiments = reactive.value([])

# initial params

experiment_path = "experiments/"
experiments_df = reactive.value([])
selected_df = reactive.value([])
selected_expirements = reactive.value([])


# generate dictionaries

def to_dict(lst):
     return {i: i for i in lst}
      




app_ui = ui.page_fluid(
    ui.input_dark_mode(),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h2("Search options"),
            ui.hr(),
            "When selecting None all are selected",
            ui.input_checkbox_group(  
                "qs_strategies",  
                "Query Strategies",  
                to_dict(qs_strategies),  
            ),
            ui.input_checkbox_group(  
                "models",  
                "models",  
                to_dict(models),  
            ),
            ui.input_checkbox_group(  
                "datasets",  
                "Datasets",  
                to_dict(datasets),  
            ),
            ui.input_checkbox_group(  
                "batch_sizes",  
                "Batch sizes",  
                to_dict(batch_sizes),  
                inline=True,
            ),
            
            ui.input_action_button("action_button", "Search"),  
            
        ),
        ui.panel_main(
            ui.output_ui("rows"),
            ui.output_data_frame("datatable"), 
        )
    ),
    ui.layout_column_wrap(
        ui.input_action_button("generate_plots", "Generate plots"),
        ui.output_text("value"),
    ),
    ui.page_fluid(
        ui.output_plot("acc_plot"),
        ui.output_plot("auroc_plot"),
    )

    
)

# TODO: load selected chenbox with query_stragies = df['qs_strategy'].unique() ..

def server(input, output, session):

    @reactive.event(input.generate_plots)
    def load_selected_df():
        expe_df = experiments_df.get().values
        selected_dataframes_list = expe_df[list(selected_experiments.get())]
        selected_expirements.set(selected_dataframes_list)
        df = []
        for selected_dataframe in selected_dataframes_list:
            filepath = Path(__file__).parent / experiment_path
            filepath = filepath.__str__() + "\\" + "\\".join(selected_dataframe)
            df.append(pd.read_csv(filepath))
        selected_df.set(df)
        

    # load selected dataframes
    @reactive.event(input.action_button)
    def load_df():
        selected_datasets_list = input.datasets()
        selected_models_list = input.models()
        selected_qs_list = input.qs_strategies()
        selected_batch_sizes_list = input.batch_sizes()
        filepath = Path(__file__).parent / "experiments.csv"
        df = pd.read_csv(filepath, dtype=str)
        if len(selected_datasets_list) > 0:
            df = df.loc[df['dataset'].isin(selected_datasets_list)]
        if len(selected_models_list) > 0:
            df = df.loc[df['model'].isin(selected_models_list)]
        if len(selected_qs_list) > 0:
            df = df.loc[df['qs_strategy'].isin(selected_qs_list)]
        if len(selected_batch_sizes_list) > 0:
            df = df.loc[df['batch_size'].isin(selected_batch_sizes_list)]
            
        

        experiments_df.set(df)

    @render.text
    def value():
        # return ", ".join(input.qs_strategies()) , ", ".join(input.models())
        return "Please load the Experiments before generating plots"
    
    @render.data_frame
    @reactive.event(input.action_button)
    def datatable():
        
        load_df()

        return render.DataTable(experiments_df.get(), selection_mode="rows") 
    
    @render.ui
    def rows():
        rows = datatable.cell_selection()["rows"]  
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        selected_experiments.set(rows)
        return f"Rows selected (select multiple with ctrl): {selected} "
    
    @render.plot(alt="Accuracy graph")
    @reactive.event(input.generate_plots)
    def acc_plot(): 
        load_selected_df()
        fig, ax = plt.subplots()
        sel_exp = selected_expirements.get()
        ax.set_title("Accuracy graph")
        ax.set_xlabel("cycles")
        ax.set_ylabel("accuracy")

        for df in selected_df.get():
            accuracy = df["accuracy"]
            ax.plot(accuracy)
        ax.legend(sel_exp[:,2])

        return fig  

    @render.plot(alt="Auroc graph")
    @reactive.event(input.generate_plots)
    def auroc_plot(): 
        fig, ax = plt.subplots()
        sel_exp = selected_expirements.get()
        ax.set_title("Accuracy graph")
        ax.set_xlabel("cycles")
        ax.set_ylabel("auroc")

        for df in selected_df.get():
            accuracy = df["auroc"]
            ax.plot(accuracy)
        ax.legend(sel_exp[:,2])

        return fig  


app = App(ui=app_ui, server=server)