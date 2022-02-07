from operator import index
import os
import argparse
import streamlit as st

from train import train
from export import export
from infer import inference
from typing import Any, Dict, Tuple
from models import available_models, desc_model
from tools.utils import open_conf, parse_metrics
from data import available_datasets, dataset_picker, desc_dataset


def create_folders() -> None:
    """Check if subfolders already exist."""
    if os.path.exists("tb_logs") is False:
        os.makedirs("tb_logs", exist_ok=True)

    if os.path.exists("exports") is False:
        os.makedirs("exports", exist_ok=True)


def sidebar(args: argparse.Namespace, conf: Dict) -> argparse.Namespace:
    """Sidebar logic is described in this method.

    Args:
        args (argparse.Namespace): argparse namespace with all
        the parameters needed for the tasks.

    Returns:
        argparse.Namespace: Updated namespace with the new parameters.
    """
    st.sidebar.title("Pytorch Lightning ⚡")

    # Select GPUs
    st.sidebar.subheader("GPUs")
    gpus_available = st.sidebar.checkbox("Are GPUs available?", value=True)
    if gpus_available:
        args.gpus = st.sidebar.number_input("Number of GPUs to use", value=1, step=1)

    st.sidebar.subheader("Training")

    # Max epochs & learning rate
    args.max_epochs = st.sidebar.number_input("Max num of epochs", value=36, step=1)
    args.auto_lr_find = st.sidebar.checkbox("Find optimal initial Learning Rate?")

    args.learning_rate = st.sidebar.number_input(
        "Learning Rate", value=3e-3, step=1e-5, format="%e"
    )

    # Batch size & workers
    args.batch_size = st.sidebar.number_input("Batch size", value=16, step=1)
    args.workers = st.sidebar.number_input("Workers", value=4, step=1)

    st.sidebar.subheader("Debug")
    if st.sidebar.checkbox("Enable fast development run over n batches"):
        args.fast_dev_run = st.sidebar.number_input("Num iterations", value=2, step=1)

    # Metrics
    st.sidebar.subheader("Metrics")
    metrics = st.sidebar.multiselect("Metrics to use", conf["metrics"])
    metrics = [parse_metrics(m) for m in metrics]
    args.metrics = " ".join(metrics)

    # Loggers
    st.sidebar.subheader("Logger")
    args.loggers = st.sidebar.selectbox("How to log metrics?", conf["loggers"])

    return args


def data_source(args: argparse.Namespace, n: int = 1) -> argparse.Namespace:
    """All the logic to fetch the data for the next steps.

    Args:
        args (argparse.Namespace): namespace with the parameters
        already defined.
        n (int, optional): Just the index. Defaults to 1.

    Raises:
        ValueError: When it tries to plot data and the DataFrame
        is empty.

    Returns:
        argparse.Namespace: Updated namespace with the new parameters.
    """
    st.subheader(f"{n}. Data source! 🤺")
    st.markdown("Select one of the datasets that are available")

    args.dataset = st.selectbox("Datasets", available_datasets())
    st.markdown(desc_dataset(args.dataset))

    return args


def model_selector(
    task: str, args: argparse.Namespace, n: int = 1
) -> argparse.Namespace:
    """Logic to select the model.

    Args:
        task (str): task selected before.
        args (argparse.Namespace): namespace with the arguments
        already selected.
        n (int, optional): Index number. Defaults to 2.

    Returns:
        argparse.Namespace: updated namespace.
    """
    st.subheader(f"{n}. Model selector! 🏗️")
    st.markdown("There is a list of models that can be selected to give them a try")

    check = True if task.lower() not in ["train"] else False
    use_check = st.checkbox("Load from a checkpoint", value=check)
    if use_check:
        col1, col2 = st.columns(2)
        st.markdown("Select a checkpoint.")
        checkp_models = ["-"] + os.listdir("tb_logs/")
        sel_model = col1.selectbox("Select model", checkp_models)

        # Select a checkpoint to start from
        checkp_mods = ["-"]
        if sel_model != "-":
            checkp_mods += os.listdir(f"tb_logs/{sel_model}")
        sel_data = col2.selectbox("Select dataset", checkp_mods)

        # Store the variable checkpoint
        if sel_data != "-" and sel_model != "-":
            args.checkpoint = os.path.join("tb_logs", sel_model, sel_data)
    else:
        args.model = st.selectbox("Models", available_models(), index=1)
        st.markdown(desc_model(args.model))

    return args


def model_hyper(args: argparse.Namespace, n: int = 1) -> argparse.Namespace:
    """Selects config parameters for the model.

    Args:
        args (argparse.Namespace): namespace with the arguments
        already selected.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        argparse.Namespace: Updated namespace.
    """
    st.subheader(f"{n}. Model hyperparameters! 💫")
    args.tokenizer = st.text_input(
        "Input a tokenizer from Hugging Face!", value="bert-base-cased"
    )

    return args


def pick_task(conf: Dict, n: int = 1) -> str:
    """Select which task to perform.

    Args:
        conf (Dict): JSON with the config parameters.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        str: task selected.
    """
    st.subheader(f"{n}. Task! 📝")
    st.markdown("It is time to select which task to perform.")
    task = st.selectbox("Tasks supported", conf["tasks"])

    return task


def run_task(task: str, args: argparse.Namespace, n: int = 1) -> Any:
    """Run the selected task! It can be: to train a new model, to test
    an already trained one, to export it or to do inference.

    Args:
        task (str): task selected before.
        args (argparse.Namespace): arguments for that task.
        n (int, optional): Index value. Defaults to 1.

    Returns:
        Any: Metrics in case of train and test. Prediction in case
        of inference and the model path in case of export.
    """
    st.subheader(f"{n}. Run! 🧟")
    task_runned = st.button(f"Launch {task.lower()}")
    try:
        if task_runned and task.lower() == "train":
            return train(args, is_st=True)
        # elif task_runned and task.lower() == "test":
        #     return test(args, is_st=True)
        elif task_runned and task.lower() == "inference":
            return inference(args, is_st=True)
        elif task_runned and task.lower() == "export":
            return export(args)
    except ValueError as ve:
        st.error(ve)


def print_metrics(metrics: Dict, n: int = 1) -> None:
    """Show the metrics (like R2 Score or Recall) from the task that
    has been run and the plot of the data fetched.

    Args:
        metrics (Tuple): Results of the metrics tracked.
        n (int, optional): Index value. Defaults to 1.
    """
    st.subheader(f"{n}. Metrics! 🗿")

    st.markdown("Lets see some results!")
    for key, met in metrics.items():
        st.markdown(f"**{key}** metrics:")
        if len(met) > 0:
            cols = st.columns(len(met))
            for idx, (key, val) in enumerate(met.items()):
                cols[idx].metric(parse_metrics(key), round(float(val), 4))


def export_model(
    args: argparse.Namespace, conf: Dict, n: int = 1
) -> argparse.Namespace:
    """Select the checkpoint to export to a ONNX or TorchScript model.

    Args:
        args (argparse.Namespace): arguments selected.
        conf (Dict): config parameters.
        n (int, optional): index values. Defaults to 1.

    Returns:
        argparse.Namespace: updated arguments
    """
    st.subheader(f"{n}. Save model! 💾")
    st.markdown(
        "Export and store the model to ONNX! In the inference task, exports can be tested."
    )

    args.name = st.text_input("File name")

    return args


def download_model(conf: Dict, n: int = 1) -> None:
    """Download a model based on the ones already exported.

    Args:
        conf (Dict): config parameters
        n (int, optional): index value. Defaults to 1.
    """
    st.subheader(f"{n}. Download the model! 📥")
    col1, col2 = st.columns(2)

    st.markdown("Select a exported mode.")
    export_mode = ["-"] + conf["exports"]
    sel_export = col1.selectbox("Select mode", export_mode)

    # Select a model
    export_files = ["-"]
    if sel_export != "-":
        export_files += os.listdir(f"exports/{sel_export.lower()}")
    sel_model = col2.selectbox("Select file", export_files)

    if sel_export != "-" and sel_model != "-":
        type = sel_export.lower()
        path = os.path.join("exports", type, sel_model)
    else:
        path = ""

    if path != "":
        file_name = path.split("/")[-1]
        with open(path, "rb") as file:
            st.download_button(
                "Download",
                data=file,
                file_name=file_name,
                mime="application/octet-stream",
            )


def inference_selector(
    args: argparse.Namespace, conf: Dict, n: int = 1
) -> argparse.Namespace:
    """Selects an exported model to do inference.

    Args:
        args (argparse.Namespace): arguments selected
        conf (Dict): config parameters
        n (int, optional): index value. Defaults to 1.

    Returns:
        argparse.Namespace: updated arguments.
    """
    st.subheader(f"{n}. Exported models! 👷")

    st.markdown("Select an exported model.")
    # Select a model
    export_files = ["-"] + os.listdir("exports/")
    sel_model = st.selectbox("Select model", export_files)
    args.model = os.path.join("exports", sel_model)

    return args


def print_prediction(value: float, n: int = 1) -> None:
    """Prints the prediction metric.

    Args:
        value (float): prediction value
        n (int, optional): index value. Defaults to 1.
    """
    st.subheader(f"{n}. Inference result! 🤖")
    st.markdown("And the result is...")
    st.metric("Predicted", float(value))


def main():
    """All the Streamlit logic is called from this method."""
    # Argparse Namespace will store all the variables to use the modules.
    args = argparse.Namespace()

    # Get config from JSON.
    conf = open_conf("conf/conf.json")["streamlit"]

    # Check if the needed folders exist
    create_folders()

    # Counter for the indexes
    n = 1

    # Set page title and favicon.
    st.set_page_config(
        page_title="TweesentGym", page_icon="🏋️", initial_sidebar_state="auto"
    )

    # Set main title
    st.title("Tweesent Gym 🏋️")
    st.markdown("Train your AI sentiment classifier for Tweesent!")

    # Sidebar
    args = sidebar(args, conf)

    # Select the task
    task = pick_task(conf, n)
    n += 1

    # Data source subheader
    if task.lower() in ["train", "test", "inference"]:
        args = data_source(args, n)
        n += 1

    # Model selector subheader
    if task.lower() != "inference":
        args = model_selector(task, args, n)
    else:
        args = inference_selector(args, conf, n)
    n += 1

    if task.lower() in ["train"] and "checkpoint" not in args:
        # Model parameters subheader
        args = model_hyper(args, n)
        n += 1

    if task.lower() == "export":
        export_model(args, conf, n)
        n += 1

    # Task subheader
    vals = run_task(task, args, n)
    if vals is not None:
        st.success(f"{task} done!")
    n += 1

    if task.lower() in ["train", "test"] and vals is not None:
        print_metrics(vals, n)
        n += 1

    # if task.lower() == "inference" and vals is not None:
    #     print_prediction(vals, n)
    #     n += 1

    if task.lower() == "export":
        download_model(conf, n)
        n += 1


if __name__ == "__main__":
    main()
