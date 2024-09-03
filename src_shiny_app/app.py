from shiny import ui, render, App, reactive
from shinywidgets import render_plotly, output_widget, render_widget  
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import urllib3
import copy
from io import StringIO
# Links to the dataset download or funktion used
DATASETLINKS = {
    "ag_news_bert-base-uncased": "https://huggingface.co/datasets/fancyzhx/ag_news",
    "aloi": "https://www.openml.org/search?type=data&status=active&id=42396",
    "banking77_bert-base-uncased": "https://huggingface.co/datasets/PolyAI/banking77",
    "cat_and_dog_dinov2_vits14": "https://huggingface.co/datasets/microsoft/cats_vs_dogs",
    "cifar100_dinov2_vits14": "https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100",
    "cifar10_dinov2_vits14": "https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10",
    "dbpedia_bert-base-uncased": "https://huggingface.co/datasets/fancyzhx/dbpedia_14",
    "dtd_dinov2_vits14": "https://pytorch.org/vision/stable/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD",
    "iris": "https://www.openml.org/search?type=data&status=active&id=61",
    "letter": "https://www.openml.org/search?type=data&status=active&id=6",
    "pendigits": "https://www.openml.org/search?type=data&status=active&id=32",
    "trec6_bert-base-uncased": "https://huggingface.co/datasets/CogComp/trec",
}
# Links to the query strategy examples
QSLINKS = {
    "Alce": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-CostEmbeddingAL-Active_Learning_with_Cost_Embedding_%28ALCE%29.html",
    "Badge": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-Badge-Batch_Active_Learning_by_Diverse_Gradient_Embedding_%28BADGE%29.html",
    "Clue": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-Clue-Clustering_Uncertainty-weighted_Embeddings_%28CLUE%29.html",
    "ContrastiveAL": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ContrastiveAL-Contrastive_Active_Learning_%28CAL%29.html",
    "CoreSet": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-CoreSet-Core_Set.html",
    "GreedySamplingX": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-GreedySamplingX-Greedy_Sampling_on_the_Feature_Space_%28GSx%29.html",
    "ProbCover": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ProbCover-Probability_Coverage_%28ProbCover%29.html",
    "ProbabilisticAL": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-ProbabilisticAL-Multi-class_Probabilistic_Active_Learning_%28McPAL%29.html",
    "RandomSampling": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-RandomSampling-Random_Sampling.html",
    "TypiClust": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-TypiClust-Typical_Clustering_%28TypiClust%29.html",
    "USEntropy": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Entropy.html",
    "USMargin": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Margin.html",
    "USLeastConfident": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/sphinx_gallery_examples/pool/plot-UncertaintySampling-Uncertainty_Sampling_with_Least-Confidence.html",
}
# Links to the models
MODELLINKS = {
    "LogisticRegression": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
    "ParzenWindowClassifier": "https://scikit-activeml.github.io/scikit-activeml-docs/latest/generated/api/skactiveml.classifier.ParzenWindowClassifier.html#skactiveml.classifier.ParzenWindowClassifier",
    "RandomForestClassifier": "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
}

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
tmp = reactive.value("")
selected_experiments = reactive.value([])

# generate dictionaries

def to_dict(lst):
    """As the checkbox ui element needs a dictonary convert the lists into dictionaries"""
    if lst[0] in DATASETLINKS.keys():
        return {i: ui.tags.a(i, href=DATASETLINKS[i], target='_blank', _add_ws=True) for i in lst}
        # return {i: ui.div(i + " ", ui.tags.a("[Link]", href=DATASETLINKS[i], target='_blank', _add_ws=True)) for i in lst}
    elif lst[0] in MODELLINKS.keys():
        return {i: ui.tags.a(i, href=MODELLINKS[i], target='_blank', _add_ws=True) for i in lst}
        # return {i: ui.div(i + " ", ui.tags.a("[Link]", href=MODELLINKS[i], target='_blank', _add_ws=True)) for i in lst}
    elif lst[0] in QSLINKS.keys():
        return {i: ui.tags.a(i, href=QSLINKS[i], target='_blank', _add_ws=True) for i in lst}
        # return {i: ui.div(i + " ", ui.tags.a("[Link]", href=QSLINKS[i], target='_blank', _add_ws=True)) for i in lst}
    else:
        return {i: i for i in lst}
      
# The app ui functions as the frontpage and includes all input and output Ui ellements
app_ui = ui.page_fluid(
    ui.input_dark_mode(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h2("Search options"),
            ui.hr(),
            "If no modalities are chosen in a category, all available options are automatically selected.",
            ui.accordion(
                ui.accordion_panel(
                    "Datasets",
                    ui.input_checkbox_group(  
                        "datasets",  
                        "",  
                        to_dict(datasets),
                        # inline=True
                    ),
                ),
                ui.accordion_panel(
                    "Models",
                    ui.input_checkbox_group(  
                        "models",  
                        "",  
                        to_dict(models),  
                        # inline=True
                    ),
                ),
                ui.accordion_panel(
                    "Query Strategies",
                    ui.input_checkbox_group(  
                        "qs_strategies",  
                        "",  
                        to_dict(qs_strategies),
                        # inline=True
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
            width=380,
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
                df.append(([], selected_dataframe[:4]))
            df[-1][0].append(pd.read_csv(StringIO(tmp.data.decode("utf-8")), index_col="step"))

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

        df_ = df[["dataset", "model", "qs_strategy", "batch_size"]].drop_duplicates()

        experiments_df.set(df_)

    @render.text
    def value():
        return "Please load the Experiments before generating plots.\n" + " Currently Loaded : " + str(len(selected_df.get())) + tmp.get()
    
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
        # This results in an error when generating the plots 
        # if len(rows) == 0:
        #     r_len = len(datatable.data_view())
        #     rows = [*range(r_len)]
        #     selected = "All"
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
        if y_label is None:
            y_label = metric_str
        sel_exp = selected_df.get()
        legend_params = []
        title_params = []
        title = metric_str
        sel_exp_names = np.array([tmp[1] for tmp in sel_exp])
        if len(sel_exp_names) <= 0:
            if use_pl:
                fig = go.Figure()
            else:
                fig, ax = plt.subplots()
                ax.set_title(f"Graph for {title}")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            return fig
        for i in range(4):
            params = np.unique(sel_exp_names[:,i])
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
        for (df_list, sel) in selected_df.get():
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
        for (df_list, sel) in selected_df.get():
            for i, df in enumerate(df_list):
            # df_list["experiment_path"] = sel
                csv = df.to_csv(index=False)
                yield ",".join(sel) + f",{i}\n"
                yield csv



app = App(ui=app_ui, server=server)