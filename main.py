from flask import Flask, render_template, request, redirect, url_for, flash,session
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import re
from datetime import datetime



app = Flask(__name__)
app.secret_key = "dev"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PLOT_FOLDER = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    session.clear()  # or selectively clear:
    session.pop("qa_archive", None)
    session.pop("insights", None)
    return render_template("pg1.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("No file selected!")
        return redirect(url_for("home"))
    
    file = request.files["file"]
    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in [".csv", ".xls", ".xlsx"]:
        flash("Unsupported file format. Please upload .csv, .xls, or .xlsx.")
        return redirect(url_for("home"))

    filename = f"{uuid.uuid4()}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    session["filepath"] = filename

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error reading uploaded file: {e}")
        return redirect(url_for("home"))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    return render_template("feature_selector.html",
                       filepath=filename,
                       numeric_columns=numeric_cols,
                       all_columns=all_cols)


@app.route("/feature_selector", methods=["POST"])
def feature_selector():
    action = request.form.get("action")
    filepath = request.form.get("filepath")
    file_path = os.path.join(UPLOAD_FOLDER, filepath)
    ext = os.path.splitext(file_path)[1].lower()


    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error reading uploaded file: {e}")
        return redirect(url_for("home"))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    if not action:
        
        return render_template("feature_selector.html",
                               filepath=filepath,
                               numeric_columns=numeric_cols,
                               all_columns=all_cols)

    if action == "describe":
        summary_html = df.describe().to_html(classes="table table-bordered", border=0)
        return render_template("result.html",
                               title="Summary Statistics",
                               table=summary_html,
                               plot_url=None,
                               filepath=filepath)

    elif action == "corr":
        try:
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            plot_filename = f"{uuid.uuid4()}.png"
            plot_path = os.path.join(PLOT_FOLDER, plot_filename)
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
            return render_template("result.html",
                                   title="Correlation Heatmap",
                                   table=None,
                                   plot_url=url_for("static", filename=f"plots/{plot_filename}"),
                                   filepath=filepath)
        except Exception as e:
            flash(f"Error generating heatmap: {e}")
            return redirect(url_for("home"))

    else:
        flash("Invalid action selected.")
        return redirect(url_for("home"))
















#INSIGHT GENERATOR AND CHATBOT
@app.route("/ask_me",methods=["GET"])
def ask_me():

    filepath = session.get("filepath")
    if not filepath:
        flash("No dataset found. Please upload a file first.")
        return redirect(url_for("home"))
    
    try:
        file_path = os.path.join(UPLOAD_FOLDER,"tmp.csv")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error loading file for form: {e}")
        return redirect(url_for("home"))
    
    if "qa_archive" not in session:
        session["qa_archive"] = []

    insights = generate_insights(df)

    return render_template("ask_me.html",
                           insights=insights,
                           archive=session["qa_archive"],
                           filepath=filepath)


def generate_insights(df):
    insights = []
     
    rows, cols = df.shape
    insights.append(f"Your data has {rows} rows and {cols} columns.")
     
    # Missing values
    null_ratios = df.isnull().mean()
    for col, ratio in null_ratios.items():
        if ratio > 0:
            percent = round(ratio * 100, 1)
            insights.append(f"Column '{col}' has {percent}% missing values.")

    # Unique values
    for col in df.columns:
        if df[col].nunique() == len(df):
            insights.append(f"Column '{col}' has all unique values (possible ID or index).")

    # Top values for categorical/text columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        top_values = df[col].value_counts().head(3)
        top_summary = ", ".join([f"{idx} ({cnt})" for idx, cnt in top_values.items()])
        insights.append(f"Column '{col}' top values: {top_summary}")

    # Datetime insights
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            min_date = df[col].min()
            max_date = df[col].max()
            insights.append(f"Column '{col}' ranges from {min_date.date()} to {max_date.date()}.")

    return insights

date_pattern = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")
def date_extractor(tokens):
    dates = []
    for token in tokens:
        if date_pattern.fullmatch(token):
            try:
                parsed = datetime.strptime(token,"%d-%m-%Y")
                dates.append(parsed.strftime("%d-%m-%Y"))
            except ValueError:
                continue
    return dates

def preprocess_q(question):
    
    tokens = question.split()
    dates = date_extractor(tokens)
    
    clean = []
    for token in tokens:
        if token in dates:
            print(token)
            clean.append(token)
        else:
            strp = "".join([char for char in token if char not in string.punctuation])
            if strp:
                clean.append(strp)
    
    stop_words = set(stopwords.words("english"))
    filtered = [word.lower() for word in clean if word.lower() not in stop_words and word not in dates]
    ints = [int(word) for word in filtered if word.isdigit()]
    keywords = [w for w in filtered if not w.isdigit()]
    dates = [datetime.strptime(d, "%d-%m-%Y").strftime("%Y-%m-%d") for d in dates]

    return dates,ints,keywords



@app.route("/process_question",methods=["POST"])
def process_question():

    question = request.form.get("question")
    dates,ints,keywords = preprocess_q(question) #date,int values can be used if needed
    
    filepath = session.get("filepath")
    file_path = os.path.join(UPLOAD_FOLDER, filepath)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path,encoding="utf-8")
    else:
        df = pd.read_excel(file_path, engine="openpyxl")

    # Default answer
    answer = "I'm not sure how to answer that. Try asking something like 'top sale' or 'average sales by product'."

    # Recognized pattern
    if "top" in keywords and "sale" in keywords:
        try:
            value_col = next((col for col in df.columns if "sale" in col.lower()), None)
            if value_col:
                result = df[value_col].max()
                answer = f"The top sale value in <strong>{value_col}</strong> is: <strong>{result}</strong>"
            else:
                answer = "No column related to 'sale' was found."
        except Exception as e:
            answer = f"Error: {str(e)}"

    # Safely retrieve and update session
    archive = session.get("qa_archive", [])
    archive.append({
        "id": len(archive),
        "question": question,
        "answer": answer
    })
    session["qa_archive"] = archive  # REASSIGNING ensures persistence



    qa_id = len(session["qa_archive"]) - 1
    return redirect(url_for("qa_detail", id=qa_id))


@app.route("/qa/<int:id>")
def qa_detail(id):
    archive = session.get("qa_archive")
    
    
    if not archive:
        flash("No archive found. Please ask a question first.")
        return redirect(url_for("ask_me"))

    if id >= len(archive):
        flash("Invalid question ID.")
        return redirect(url_for("ask_me"))

    qa = archive[id]
    return render_template("qa_detail.html", question=qa["question"], answer=qa["answer"])




###########################################################################

















#TIMESERIES
@app.route("/timeseries_form",methods=["GET"])
def timeseries_form():
    try:
        file_path = os.path.join(UPLOAD_FOLDER,"tmp.csv")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error loading file for form: {e}")
        return redirect(url_for("home"))
    
    numeric_cols = df.select_dtypes(include="number").columns.to_list()
    all_cols = df.columns.tolist()

    return render_template("timeseries_form.html",
                           numeric_columns = numeric_cols,
                           all_columns = all_cols,
                           filepath="tmp.csv")

   
@app.route("/render_timeseries", methods=["POST"])
def render_timeseries():
    filepath = request.form.get("filepath")
    file_path = os.path.join(UPLOAD_FOLDER,filepath)
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Could not reload Excel file: {e}")
        return redirect(url_for("home"))
    
    date_col = request.form.get("date_col")
    value_col = request.form.get("value_col")
    forecast_horizon = int(request.form.get("forecast_horizon"))

    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    except Exception as e:
        flash(f"The selected column '{date_col}' could not be parsed as dates: {e}")
        return redirect(url_for("render_timeseries")) 
    
    num_lags = 3
    df.sort_values(by=date_col)
    df = df[[date_col, value_col]].copy()
    
    for lag in range(1, num_lags + 1):
        df[f"lag_{lag}"] = df[value_col].shift(lag)

    df.dropna(inplace=True)

    model = RandomForestRegressor(n_estimators=200,random_state=42)
    lag_cols  = [f"lag_{i}" for i in range(1, num_lags + 1)]
    x = df[lag_cols]
    y = df[value_col]
    model.fit(x,y)

    last_known_date  = df[date_col].iloc[-1]
    last_row_lags    = df[lag_cols].iloc[-1].values.tolist()
    
    future_dates = []
    forecast_values = []

    for step in range(forecast_horizon):
        # 2a.    Convert current lag list to 2-D array → model expects shape (1, num_lags)
        X_next = np.array(last_row_lags).reshape(1, -1)

        # 2b.    Predict next value
        y_next = model.predict(X_next)[0]      # scalar

        # 2c.    Append prediction and future date to our lists
        next_date = last_known_date + pd.Timedelta(1, unit="D")   # adjust 'unit' if weekly/monthly
        future_dates.append(next_date)
        forecast_values.append(y_next)

        # 2d.    Update 'last_row_lags' for next iteration
        last_row_lags = [y_next] + last_row_lags[:-1]   # insert at front, drop oldest

        # 2e.    Advance the reference date
        last_known_date = next_date


    try: 

        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(df[date_col], df[value_col], label="Actual")
        ax.plot(future_dates, forecast_values, label="Forecast", linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel(value_col)
        ax.legend()

        plot_filename = f"{uuid.uuid4()}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
   
        return render_template("result.html",
                               title=f"Histrocial vs. Predictions",
                               table=None,
                               plot_url=url_for("static", filename=f"plots/{plot_filename}"),
                               filepath=filepath)

    except Exception as e:
        flash(f"Failed to generate predictions: {e}")
        return redirect(url_for("home"))


#######################################











#JOINTPLOT

@app.route("/jointplot_form", methods=["GET"])
def jointplot_form():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "tmp.csv")  # or tmp.xlsx
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error loading file for form: {e}")
        return redirect(url_for("home"))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    return render_template("jointplot_form.html",
                           numeric_columns=numeric_cols,
                           all_columns=all_cols,
                           palettes=["deep", "muted", "pastel", "bright", "dark", "colorblind"],
                           filepath="tmp.csv")



@app.route("/render_jointplot", methods=["POST"])
def render_jointplot():
    filepath = request.form.get("filepath")
    file_path = os.path.join(UPLOAD_FOLDER, filepath)
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Could not reload Excel file: {e}")
        return redirect(url_for("home"))
    
    x_axis = request.form.get("x_axis")
    y_axis = request.form.get("y_axis")
    hue = request.form.get("hue") or None
    color = request.form.get("color")
    kind = request.form.get("kind")

    try:
        plt.clf()
        if kind=="hex":
            flash("The kind hex is not supported with hue, so the hue is removed")
            g = sns.jointplot(data=df,x=x_axis,y=y_axis,kind="hex",color=color)
        else:
            g = sns.jointplot(data=df,x=x_axis,y=y_axis,kind=kind,color=color,hue=hue)
            
        g.set_axis_labels(x_axis,y_axis)

        plot_filename = f"{uuid.uuid4()}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        g.figure.savefig(plot_path,bbox_inches="tight")
        plt.close(g.figure)

        return render_template("result.html",
                           title=f"Jointplot of {x_axis} and {y_axis}",
                           table=None,
                           plot_url=url_for("static", filename=f"plots/{plot_filename}"),
                           filepath=filepath)
    
    except Exception as e:
        flash(f"Failed to generate jointplot: {e}")
        return redirect(url_for("home"))


#################################################################3



#HISTOGRAM

@app.route("/render_histogram", methods=["POST"])
def render_histogram():
    filepath = request.form.get("filepath")
    file_path = os.path.join(UPLOAD_FOLDER, filepath)
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Could not reload Excel file: {e}")
        return redirect(url_for("home"))

    column = request.form.get("hist_column")
    hue = request.form.get("hist_hue") or None
    bins = int(request.form.get("hist_bins") or 20)
    color = request.form.get("hist_color")

    try:
        plt.clf()
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=column, hue=hue, bins=bins, color=color, ax=ax)

        plot_filename = f"{uuid.uuid4()}.png"
        plot_path = os.path.join(PLOT_FOLDER, plot_filename)
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)

        return render_template("result.html",
                               title=f"Histogram of {column}",
                               table=None,
                               plot_url=url_for("static", filename=f"plots/{plot_filename}"),
                               filepath=filepath)
    except Exception as e:
        flash(f"Failed to generate histogram: {e}")
        return redirect(url_for("home"))

@app.route("/histogram_form", methods=["GET"])
def histogram_form():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, "tmp.csv")  # or tmp.xlsx
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        flash(f"Error loading file for form: {e}")
        return redirect(url_for("home"))

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    return render_template("histogram_form.html",
                           numeric_columns=numeric_cols,
                           all_columns=all_cols,
                           palettes=["deep", "muted", "pastel", "bright", "dark", "colorblind"],
                           filepath="tmp.csv")

#########################################################33







if __name__ == "__main__":
    app.run(debug=True)