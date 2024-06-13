from shiny import ui, render, App, reactive
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import mlflow
import pandas as pd
import os

# available selections

qs_strategies = ["random", "uncertainty_sampling", "probabilistic_active_learning"]
models = ["logistic regression", "parzen window classifier", "random forest tree"]
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

# initial params


df = reactive.value([])
filepath = reactive.value("")


# generate dictionaries

def to_dict(lst):
     return {i: i for i in lst}
      




app_ui = ui.page_fluid(
    ui.input_dark_mode(),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.h2("Search options"),
            ui.hr(),
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
            ui.output_data_frame("datatable"), 
        )
    ),
    ui.layout_columns(
        ui.output_text("value"),
    )
    
)

def server(input, output, session):
    # load selected dataframes
    @reactive.event(input.action_button)
    def load_df():
            filepath = Path(__file__).parent / "result.csv"
            df = pd.read_csv(filepath)
            return df

    @render.text
    def value():
         return ", ".join(input.qs_strategies()) , ", ".join(input.models())
    
    @render.data_frame
    @reactive.event(input.action_button)
    def datatable():
        selected_datasets_list = input.datasets()
        selected_models_list = input.models()
        selected_qs_list = input.qs_strategies()
        selected_batch_sizes_list = input.batch_sizes()
        df = load_df()

        return render.DataTable(df) 


app = App(ui=app_ui, server=server)