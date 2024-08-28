from shiny import ui, render, App, reactive
from shinywidgets import render_plotly, output_widget, render_widget  
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import urllib3
from io import StringIO

app_dir = Path(__file__).parent

# available selections

def load_options():
    """This function loads all available selection for each category"""
    filepath = app_dir / "experiments.csv"
    df = pd.read_csv(filepath, dtype=str)
    qs_strategies = sorted(df['qs_strategy'].unique())
    models = sorted(df['model'].unique())
    datasets = sorted(df['dataset'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    return qs_strategies, models, datasets, batch_sizes
qs_strategies, models, datasets, batch_sizes = load_options()

# initial params

experiment_path = "experiments_results\\"
experiments_df = reactive.value([])
all_experiments_df = reactive.value()
selected_df = reactive.value([])
list_expirement_metrics = reactive.value([])
selected_experiments = reactive.value([])

# generate dictionaries

def to_dict(lst):
    """As the checkbox ui element needs a dictonary convert the lists into dictionaries"""
    return {i: i for i in lst}
      
# The app ui functions as the frontpage and includes all input and output Ui ellements
app_ui = ui.page_fluid(
    ui.input_dark_mode(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h2("Search options"),
            ui.hr(),
            "When selecting no modalities in each category all available results are selected",
            ui.accordion(
                ui.accordion_panel(
                    "Datasets",
                    ui.input_checkbox_group(  
                        "datasets",  
                        "",  
                        to_dict(datasets),  
                    ),
                ),
                ui.accordion_panel(
                    "Models",
                    ui.input_checkbox_group(  
                        "models",  
                        "",  
                        to_dict(models),  
                    ),
                ),
                ui.accordion_panel(
                    "Query Strategies",
                    ui.input_checkbox_group(  
                        "qs_strategies",  
                        "",  
                        to_dict(qs_strategies),  
                    ),   
                ),
                ui.accordion_panel(
                    "Batch Size",
                    ui.input_checkbox_group(  
                        "batch_sizes",  
                        "",  
                        to_dict(batch_sizes),  
                        inline=True,
                    ),
                ),
                open=False
            ), 
            open="open",
            width=350,
        ),
        ui.input_action_button("action_button", "Search"),
        ui.output_ui("rows"),
        ui.page_fluid(
            ui.output_data_frame("datatable"),
        ),
        height="800px" 
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
        output_widget("neg_brier_score"),
        output_widget("neg_log_loss"),
        output_widget("average_precision"),
        output_widget("balanced_accuracy"),
        output_widget("time"),
        ui.download_button("download", "Download results")
    )

    
)

def server(input, output, session):

    def load_experiment():
        """Function to load the selected expirements to allow the ploting for all metrics"""
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
        http = urllib3.PoolManager(num_pools=1)
        for selected_dataframe in selected_dataframes_list:
            current_path = "/".join(selected_dataframe[:4])
            url_path = "/".join(selected_dataframe)
            # If you are using a local file change the url path with "file://path/to/file"
            csv_url = f"https://raw.githubusercontent.com/AlexanderBenz/scikit-activeml-experiments/main/experiments/experiments_results/{url_path}"
            tmp = http.request("GET", csv_url, timeout=0.2)

            if old_path != current_path:
                df.append([])
            df[-1].append(pd.read_csv(StringIO(tmp.data.decode("utf-8")), index_col="step"))

            old_path = current_path
        selected_df.set(df)

    # load selected dataframes
    @reactive.event(input.action_button)
    def load_df():
        """Function to load the selectet expirement combinations using the selectet parameters"""
        selected_datasets_list = input.datasets()
        selected_models_list = input.models()
        selected_qs_list = input.qs_strategies()
        selected_batch_sizes_list = input.batch_sizes()
        filepath = app_dir / "experiments.csv"
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
        return "Please load the Experiments before generating plots.\n" + " Currently Loaded : " + str(len(list_expirement_metrics.get()))
    
    @render.data_frame
    @reactive.event(input.action_button)
    def datatable():
        """Display the selected experiment combinations"""
        load_df()
        return render.DataTable(experiments_df.get(), selection_mode="rows", width='fit-page', height='fit-page') 
    
    
    @render.ui
    def rows():
        """Ui function to display and save the chosen rows of the experiments. """
        rows = datatable.cell_selection()["rows"]  
        selected = ", ".join(str(i) for i in sorted(rows)) if rows else "None"
        selected_experiments.set(rows)
        return f"Rows selected (select multiple with ctrl): {selected} "
    
    def create_fig(metric_str, x_label="number of samples", y_label=None, use_pl=True, use_bar=False):
        """
        Function to create plotly plots or matplotlip plots

        Parameters:
        - metric_str (str): String to select the correct metric from the selected experiments.
        - x_label (str): Name of the x cordinate.
        - y_label (str|None): Name of the y cordinate. If none use the metric_str
        - use_pl (bool): Boolean deciding the plot return. If (True) return a
        plotly plot else if (False) return a matplotlip plot
        - use_bar (bool): Boolean deciding the plotly return. If (True) return 
        a bar plot else if (False) return a scatter plot
        """
        sel_exp = list_expirement_metrics.get()
        sel_df = selected_df.get()
        if y_label is None:
            y_label = metric_str

        legend_params = []
        title_params = []
        title = metric_str
        for i in range(4):
            params = np.unique(sel_exp[:,i])
            if len(params) <= 1:
                title_params.extend(params)
            else:
                legend_params.append(i)
        if len(title_params) > 0:
            title = '+'.join(title_params)

        # Create a plotly figure of matplotlib figure
        if use_pl:
            fig = go.Figure()
        else:
            fig, ax = plt.subplots()
            ax.set_title(f"Graph for {title}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)

        # Itterate through all selected experiments and metrics
        for (df_list, sel) in zip(sel_df,sel_exp):
            # Save all metrics for each df as most experiments are run on multiple seeds
            metric = []
            for df in df_list:
                metric.append(df[f"{metric_str}"])
            # calculate the error bars 
            reshaped_result = np.array(metric).reshape((-1, len(metric[0])))
            errorbar_mean = np.mean(reshaped_result, axis=0)
            errorbar_std = np.std(reshaped_result, axis=0)
            batch_size = int(sel[3])
            label_name = f"({np.mean(errorbar_mean):.4f}) {'+'.join(sel[legend_params])}"
            if use_pl:
                if use_bar:
                    fig.add_trace(go.Bar(
                        name=f"Mean Runtime: ({np.sum(errorbar_mean):.2f}) {'+'.join(sel[legend_params])}",
                        x=['+'.join(sel[legend_params])], y=errorbar_mean,
                        error_y=dict(type='data', array=errorbar_std)
                    ))
                else:
                    fig.add_trace(go.Scatter(
                    name=label_name,
                    x=np.arange(batch_size, (len(metric[0])+1)*batch_size, step=batch_size),
                    y=errorbar_mean,
                    error_y=dict(
                        type='data', # value of error bar given in data coordinates
                        array=errorbar_std,
                        visible=True)
                        )
                    )        
            else:
                ax.errorbar(np.arange(batch_size, (len(metric[0])+1)*batch_size, step=batch_size), errorbar_mean, errorbar_std, label=label_name, alpha=0.5)
        if use_pl:
            fig.update_layout(title=dict(text=title), xaxis_title=x_label, yaxis_title=y_label)
            if use_bar:
                fig.update_layout(barmode='group')
        else:
            ax.legend()

        return fig
    # ------- Widgets to generate the plots ---------

    # @render.plot(alt="Accuracy graph")
    @render_widget
    @reactive.event(input.generate_plots)
    def acc_plot():
        load_experiment()
        return create_fig("accuracy")

    @render_widget
    @reactive.event(input.generate_plots)
    def auroc_plot():
        return create_fig("auroc")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def f1_micro(): 
        return create_fig("f1_micro")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def f1_macro(): 
        return create_fig("f1_macro")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def neg_brier_score(): 
        return create_fig("neg_brier_score")  

    @render_widget
    @reactive.event(input.generate_plots)
    def neg_log_loss(): 
        return create_fig("neg_log_loss")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def average_precision(): 
        return create_fig("average_precision")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def balanced_accuracy(): 
        return create_fig("balanced_accuracy")  
    
    @render_widget
    @reactive.event(input.generate_plots)
    def time(): 
        return create_fig("time", use_bar=True, x_label="", y_label="Time in seconds per query")  
    
    # generate the download Button
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