from shiny import ui, render, App, reactive
from shinywidgets import render_plotly, output_widget, render_widget  
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# import mlflow
import pandas as pd
import os

# available selections

qs_strategies = ["random", "UncertaintySampling", "probabilistic_active_learning"]
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

experiment_path = "experiments_results\\"
experiments_df = reactive.value([])
all_experiments_df = reactive.value()
selected_df = reactive.value([])
list_expirement_metrics = reactive.value([])


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
        output_widget("acc_plot"),
        output_widget("auroc_plot"),
        output_widget("f1_micro"),
        output_widget("f1_macro"),
        # ui.output_plot("neg_brier_score"),
        output_widget("neg_log_loss"),
        output_widget("average_precision"),
        output_widget("balanced_accuracy"),
        # output_widget("acc_plotly"),
        ui.download_button("download", "Download results")
    )

    
)

# TODO: load selected chenbox with query_stragies = df['qs_strategy'].unique() ..

def server(input, output, session):

    @reactive.event(input.generate_plots)
    def load_selected_df():
        expe_df = experiments_df.get().values
        all_exp_df = all_experiments_df.get()
        sel_dataframe = expe_df[list(selected_experiments.get())]
        selected_dataframes_list = all_exp_df.loc[all_exp_df['dataset'].isin(sel_dataframe[:,0])]
        selected_dataframes_list = selected_dataframes_list.loc[selected_dataframes_list['model'].isin(sel_dataframe[:,1])]
        selected_dataframes_list = selected_dataframes_list.loc[selected_dataframes_list['qs_strategy'].isin(sel_dataframe[:,2])]
        selected_dataframes_list = selected_dataframes_list.loc[selected_dataframes_list['batch_size'].isin(sel_dataframe[:,3])]
        selected_dataframes_list = selected_dataframes_list.values
        list_expirement_metrics.set(sel_dataframe)
        
        df = []
        old_path = ""
        for selected_dataframe in selected_dataframes_list:
            
            filespath = Path(__file__).parent / experiment_path
            current_path = "\\".join(selected_dataframe[:4])
            # If you run shiny localy raplace the + with + "/" +
            filespath = filespath.__str__() + "\\".join(selected_dataframe)
            if old_path != current_path:
                df.append([])
            df[-1].append(pd.read_csv(filespath))
            old_path = current_path
            
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
        all_experiments_df.set(df)
        df = df[["dataset", "model", "qs_strategy", "batch_size"]].drop_duplicates()

        experiments_df.set(df)

    @render.text
    def value():
        # return ", ".join(input.qs_strategies()) , ", ".join(input.models())
        return "Please load the Experiments before generating plots", len(list_expirement_metrics.get())
    
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
    
    def create_fig(metric_str, x_label="number of samples", y_label=None, use_pl=True):
        sel_exp = list_expirement_metrics.get()
        if y_label is None:
            y_label = metric_str
        if use_pl:
            fig = go.Figure()
        else:
            fig, ax = plt.subplots()
            ax.set_title(f"{metric_str} graph")
            ax.set_xlabel(x_label)
            
            ax.set_ylabel(y_label)

        for (df_list, sel) in zip(selected_df.get(),sel_exp):
            metric = []
            for df in df_list:
                metric.append(df[f"{metric_str}"])
            reshaped_result = np.array(metric).reshape((-1, len(metric[0])))
            errorbar_mean = np.mean(reshaped_result, axis=0)
            errorbar_std = np.std(reshaped_result, axis=0)
            batch_size = int(sel[3])
            label_name = f"({np.mean(errorbar_mean):.4f}) {'+'.join(sel[:4])}"
            if use_pl:
                fig.add_trace(go.Scatter(
                name=label_name,
                x=np.arange(0, (len(metric[0]))*batch_size, step=batch_size),
                y=errorbar_mean,
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=errorbar_std,
                    visible=True)
                    )
                )
                
            else:
                ax.errorbar(np.arange(0, (len(metric[0]))*batch_size, step=batch_size), errorbar_mean, errorbar_std, label=label_name, alpha=0.5)
        if use_pl:
            fig.update_layout(title=dict(text=metric_str), xaxis_title=x_label, yaxis_title=y_label)
        else:
            ax.legend()

        return fig

    # @render.plot(alt="Accuracy graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def acc_plot(): 
        load_selected_df()
        return create_fig("accuracy")

    # @render.plot(alt="Auroc graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def auroc_plot():
        return create_fig("auroc")  
    
    # @render.plot(alt="f1_micro graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def f1_micro(): 
        return create_fig("auroc")  
    
    # @render.plot(alt="f1_macro graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def f1_macro(): 
        return create_fig("f1_macro")  
    
    # @render.plot(alt="neg_brier_score graph")
    # @reactive.event(input.generate_plots)
    # def neg_brier_score(): 
    #     load_selected_df()
    #     return create_fig("neg_brier_score")  

    # @render.plot(alt="neg_log_loss graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def neg_log_loss(): 
        return create_fig("neg_log_loss")  
    
    # @render.plot(alt="average_precision graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def average_precision(): 
        return create_fig("average_precision")  
    
    # @render.plot(alt="balanced_accuracy graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def balanced_accuracy(): 
        return create_fig("balanced_accuracy")  
    
    # @render_widget
    # @reactive.event(input.generate_plots)
    # def acc_plotly(): 
    #     return create_fig("accuracy", use_pl=True)  
    
    @render.download(filename="experiemnt_results.csv")
    @reactive.event(input.generate_plots)
    #TODO: look at how to zip the files
    def download():
        sel_exp = list_expirement_metrics.get()
        for (df_list, sel) in zip(selected_df.get(),sel_exp):
            for i, df in enumerate(df_list):
            # df_list["experiment_path"] = sel
                csv = df.to_csv(index=False)
                yield ",".join(sel) + f",{i}\n"
                yield csv



app = App(ui=app_ui, server=server)