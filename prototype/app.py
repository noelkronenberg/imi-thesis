from flask import Flask
from flask import render_template, request, redirect
import json
import pandas as pd
import random

app = Flask(__name__)

# reference: 
# https://youtu.be/E2hytuQvLlE

# get data for case_id
def find_data(case_id):
    with open('noel_thesis/app/data/data.json', 'r') as file:
        data = json.load(file)
        for _, row_data in data.items():
            if row_data.get('case_id') == case_id:
                return row_data
    return None

def transform_data(data, risk_score_name, variable_names, color):
    """
    Transforms data into correct format.

    Parameters:
        data: JSON object containing the data for one case.
        risk_score_means: Means for each risk score.
        risk_score_name: Name of the risk score as a tuple (column name, display name).
        variable_names: Names of the risk score variables as a tuple [(column name, display name), (column name, display name), ...].
        color: RBG value for the color display.

    Returns:
        list: A list with the correctly transformed data.
    """
    
    risk_score_value = data.get(risk_score_name[0])

    variables = []
    total_variable_sum = 0

    # get sum of all values
    for variable_name in variable_names:
        variable_value = data.get(variable_name[0])
        total_variable_sum += variable_value

    # get values for variables
    for variable_name in variable_names:
        variable_value = data.get(variable_name[0])

        # set color in proportion to variable value
        opacity = variable_value / total_variable_sum if total_variable_sum != 0 else 0
        variable_color = f"rgba{tuple(int(x) for x in color[4:-1].split(',')) + (opacity,)}"
        
        variables.append((variable_name[1], variable_value, variable_color))

    transformed_data = [
        (risk_score_name[1], risk_score_value, color, [
            (variable_name, variable_value, variable_color) for variable_name, variable_value, variable_color in variables
        ])
    ]

    return transformed_data

def calculate_cohort_average_for_case(df:pd.DataFrame, column:str, case_id:str, age:bool=True, ops:bool=True, age_variance:int=5):
    """
    Calculate the mean value of a specified column for rows with similar values of 'age_during_op' and 'ops_code' for a given row.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to calculate the average.
        case_id (str): The case_id for which to calculate the average. Default is None.

    Returns:
        float: The mean value of the specified column for rows with similar values in 'age_during_op' and 'ops_code'
        for the given case_id.
    """

    row = df[df['case_id'] == case_id].iloc[0]

    # get values
    age = row['age_during_op']
    ops_code = row['ops_code']

    # filter rows with similar values

    # ignore by default
    same_age = True
    same_ops = True
    
    # if configured, check for parameters
    if age:
        same_age = (df['age_during_op'].between(age - age_variance, age + age_variance))

    if ops:
        same_ops = (df['ops_code'] == ops_code)

    not_self = (df['case_id'] != case_id)
    similar_rows = df[same_age & same_ops & not_self]

    # calculate mean
    cohort_mean = round(similar_rows[column].median(), 2)

    # get n
    n_comparisons = similar_rows.shape[0]

    return cohort_mean, n_comparisons

def get_random_case_id():
    with open('noel_thesis/app/data/data.json', 'r') as file:
        data = json.load(file)
        case_ids = [data[key]["case_id"] for key in data]
    return random.choice(case_ids)

@app.route('/')
def index():
    return redirect(f'/{get_random_case_id()}/?dev=true&age=true')

@app.route('/<case_id>/')
def chart(case_id):

    # check if in dev mode 
    if request.args.get('dev', 'false').lower() == 'true':
        case_id = get_random_case_id()

    df_patients_cohort= pd.read_json('noel_thesis/app/data/data.json', orient='index')
    scores = []

    # toggles for parameters
    age = request.args.get(key='age', default='false', type=str).lower() == 'true'
    ops = request.args.get(key='ops', default='false', type=str).lower() == 'true'

    # get data (hard coded)
    data = find_data(f'{case_id}')
    RCRI_data = transform_data(data, ('MACE_risk', 'Cardiac'), [('elevated_risk_surgery', 'Elevated Risk Surgery'), ('MI_history', 'MI History'), ('CHF_history', 'CHF History'), ('CD_history', 'CD History'), ('insulin', 'Insulin')], 'rgb(197, 34, 51)')
    stroke_data = transform_data(data, ('STT_risk', 'Stroke'), [('hypertension_history', 'Hypertension History'), ('CHF_history', 'CHF History'), ('diabetes_history', 'Diabetes History'), ('vascular_history', 'Vascular History'), ('STT_history', 'CVD History'), ('sex_female', 'Female'), ('STT_age_risk', 'Age')], 'rgb(197, 34, 51)')
    scores.extend(RCRI_data)
    scores.extend(stroke_data)

    # set color in proportion to score value
    total_score_value = sum(row[1] for row in scores)
    opacities = [row[1] / total_score_value for row in scores]
    for i in range(len(scores)):
        scores[i] = (scores[i][0], scores[i][1], f"rgba{tuple(int(x) for x in scores[i][2][4:-1].split(',')) + (opacities[i],)}", scores[i][3])

    # extract values (hard coded)

    RCRI_median, n_comparisons = calculate_cohort_average_for_case(df=df_patients_cohort, column='MACE_risk', case_id=case_id, age=age, ops=ops)
    stroke_median, _ = calculate_cohort_average_for_case(df=df_patients_cohort, column='STT_risk', case_id=case_id, age=age, ops=ops)
    score_median_values = [RCRI_median, stroke_median]

    score_labels = [row[0] for row in scores]
    score_values = [row[1] for row in scores]
    score_colors = [row[2] for row in scores]

    cardiac_variable_labels = [row[0] for row in scores[0][3]]
    cardiac_variable_values = [row[1] for row in scores[0][3]]
    cardiac_variable_colors = [row[2] for row in scores[0][3]]

    stroke_variable_labels = [row[0] for row in scores[1][3]]
    stroke_variable_values = [row[1] for row in scores[1][3]]
    stroke_variable_colors = [row[2] for row in scores[1][3]]

    # forward to chart (hard coded)
    return render_template(template_name_or_list='chart.html', 
                           case_id=case_id,
                           score_median_values=score_median_values, n_comparisons=n_comparisons,
                           score_labels=score_labels, scores_values=score_values, score_colors=score_colors, 
                           cardiac_variable_labels=cardiac_variable_labels, cardiac_variable_values=cardiac_variable_values, cardiac_variable_colors=cardiac_variable_colors,
                           stroke_variable_labels=stroke_variable_labels, stroke_variable_values=stroke_variable_values, stroke_variable_colors=stroke_variable_colors,
                           zip=zip
                           )

if __name__ == "__main__":
    app.run(port=8500, debug=True)