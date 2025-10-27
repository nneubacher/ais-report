import io
import time
import pandas as pd
import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import dotenv
from langchain_ibm import WatsonxLLM
import requests.exceptions
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


dotenv.load_dotenv()
WATSONX_API_KEY = os.environ.get("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID")
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

DEFAULT_MODEL_ID = "ibm/granite-3-2-8b-instruct"

if not WATSONX_API_KEY:
    print("ERROR: WATSONX_API_KEY not set.")
if not WATSONX_PROJECT_ID:
    print("ERROR: WATSONX_PROJECT_ID not set.")

app = Flask(__name__)
CORS(app)

def analyze_dataframe(df):
    insights = {}
    print("\n--- [Insight Engine] Start ---")

    # numerical collumn conversion
    numeric_cols = [
        'total_distance_nm', 'total_Fuel_consumption_kg', 'total_CO₂_emissions_kg_CO₂',
        'Total fuel consumption [m tonnes]', 'Total CO₂ emissions [m tonnes]',
        'Total CH₄ emissions [m tonnes]', 'Total N₂O emissions [m tonnes]',
        'Total CO₂eq emissions [m tonnes]', 'Time spent at sea [hours]',
        'Fuel consumption per distance [kg / n mile]',
        'CO₂ emissions per distance [kg CO₂ / n mile]',
        'CO₂eq emissions per distance [kg CO₂eq / n mile]',
        'VesselType', 'Length', 'Width', 'Draft'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Numeric column '{col}' not found.")

    print("\n--- [Insight Engine] After Numeric Conversion: ---")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    # extract technical efficiency rating
    if 'Technical efficiency' in df.columns:
        df['Efficiency_Rating_Type'] = df['Technical efficiency'].astype(str).str.split(' ').str[0].str.strip('()')
        print("\n--- [Insight Engine] Technical Rating Types Found: ---")
        print(df['Efficiency_Rating_Type'].value_counts())
    else:
        print("Warning: 'Technical efficiency' column not found.")
        df['Efficiency_Rating_Type'] = 'Unknown'

    # filter out invalid entries
    valid_df = df[df['CO₂ emissions per distance [kg CO₂ / n mile]'] > 0].copy()

    insights['total_vessels'] = len(df)
    insights['valid_efficiency_entries'] = len(valid_df)
    insights['invalid_entries'] = insights['total_vessels'] - insights['valid_efficiency_entries']

    # fleet-wide statistics
    efficiency_stats = valid_df['CO₂ emissions per distance [kg CO₂ / n mile]'].describe()
    insights['fleet_efficiency_stats'] = {
        'mean': efficiency_stats.get('mean', 0),
        'median': efficiency_stats.get('50%', 0),
        '25th_percentile': efficiency_stats.get('25%', 0),
        '75th_percentile': efficiency_stats.get('75%', 0),
        'min': efficiency_stats.get('min', 0),
        'max': efficiency_stats.get('max', 0)
    }

    # fleet activity stats
    insights['time_at_sea_stats'] = df['Time spent at sea [hours]'].describe().to_dict()
    insights['distance_stats'] = df['total_distance_nm'].describe().to_dict()

    # analysis by ship type
    avg_by_shiptype = valid_df.groupby('Ship type')['CO₂ emissions per distance [kg CO₂ / n mile]'].mean().sort_values(ascending=False)
    insights['avg_by_shiptype'] = avg_by_shiptype.to_dict()

    # analysis by technical efficiency rating
    avg_by_rating = valid_df.groupby('Efficiency_Rating_Type')['CO₂ emissions per distance [kg CO₂ / n mile]'].mean().sort_values(ascending=False)
    if not avg_by_rating.empty:
        insights['avg_by_rating'] = avg_by_rating.to_dict()

    # top/bottom performers
    top_5_efficient = valid_df.sort_values(by='CO₂ emissions per distance [kg CO₂ / n mile]', ascending=True).head(5)
    insights['top_5_efficient'] = top_5_efficient[['Name', 'Ship type', 'CO₂ emissions per distance [kg CO₂ / n mile]']].to_dict('records')

    least_efficient_df = df.copy()
    least_efficient_df['CO₂ emissions per distance [kg CO₂ / n mile]'] = least_efficient_df['CO₂ emissions per distance [kg CO₂ / n mile]'].fillna(0)
    top_5_least_efficient = least_efficient_df.sort_values(by='CO₂ emissions per distance [kg CO₂ / n mile]', ascending=False).head(5)
    insights['top_5_least_efficient'] = top_5_least_efficient[['Name', 'Ship type', 'CO₂ emissions per distance [kg CO₂ / n mile]']].to_dict('records')

    # top emitter impact (Pareto)
    df_sorted_by_total = df.sort_values(by='total_CO₂_emissions_kg_CO₂', ascending=False)
    total_fleet_emissions = df_sorted_by_total['total_CO₂_emissions_kg_CO₂'].sum()

    num_top_10_percent = int(len(df_sorted_by_total) * 0.10)
    top_10_percent_emissions = df_sorted_by_total.head(num_top_10_percent)['total_CO₂_emissions_kg_CO₂'].sum()

    if total_fleet_emissions > 0:
        percentage_impact = (top_10_percent_emissions / total_fleet_emissions) * 100
    else:
        percentage_impact = 0

    insights['pareto_analysis'] = {
        'top_10_percent_count': num_top_10_percent,
        'top_10_percent_impact_pct': percentage_impact,
        'total_fleet_emissions_kg': total_fleet_emissions
    }

    insights['correlation'] = {}

    # correlation 1: activity vs. total emissions
    corr_cols_activity = ['Time spent at sea [hours]', 'total_distance_nm', 'total_CO₂_emissions_kg_CO₂']
    if all(col in df.columns for col in corr_cols_activity):
        corr_matrix_activity = df[corr_cols_activity].corr()
        insights['correlation']['activity_vs_total_emissions'] = {
            'time_vs_total_emissions': corr_matrix_activity.loc['Time spent at sea [hours]', 'total_CO₂_emissions_kg_CO₂'],
            'distance_vs_total_emissions': corr_matrix_activity.loc['total_distance_nm', 'total_CO₂_emissions_kg_CO₂']
        }

    # correlation 2: size vs. efficiency
    corr_cols_efficiency = ['Length', 'Width', 'Draft', 'CO₂ emissions per distance [kg CO₂ / n mile]']
    if all(col in valid_df.columns for col in corr_cols_efficiency):
        corr_matrix_efficiency = valid_df[corr_cols_efficiency].dropna().corr()
        if not corr_matrix_efficiency.empty:
            insights['correlation']['size_vs_efficiency'] = {
                'length_vs_efficiency': corr_matrix_efficiency.loc['Length', 'CO₂ emissions per distance [kg CO₂ / n mile]'],
                'width_vs_efficiency': corr_matrix_efficiency.loc['Width', 'CO₂ emissions per distance [kg CO₂ / n mile]'],
                'draft_vs_efficiency': corr_matrix_efficiency.loc['Draft', 'CO₂ emissions per distance [kg CO₂ / n mile]']
            }

    print("\n--- [Insight Engine] Analysis Complete ---")
    return insights

def generate_prompt_from_insights(insights):

    def format_dict_list(title, data_list, value_key):
        s = f"[{title}]\n"
        if not data_list:
            return f"[{title}]\n- (No data available)\n"
        for i, item in enumerate(data_list, 1):
            s += f"{i}. {item['Name']} ({item['Ship type']}): {item.get(value_key, 0):,.2f} kg/nm\n"
        return s

    def format_dict_averages(title, data_dict):
        s = f"[{title}]\n"
        if not data_dict:
            return f"[{title}]\n- (No data available)\n"
        for key, avg in data_dict.items():
            s += f"- {key}: {avg:,.2f} kg/nm\n"
        return s

    avg_by_shiptype_str = format_dict_averages("Efficiency by Ship Type (Avg CO2/nm)", insights.get('avg_by_shiptype', {}))
    avg_by_rating_str = format_dict_averages("Efficiency by Technical Rating (Avg CO2/nm)", insights.get('avg_by_rating', {}))
    top_5_efficient_str = format_dict_list("Top 5 Most Efficient Vessels (Lowest CO2/nm)", insights.get('top_5_efficient', []), 'CO₂ emissions per distance [kg CO₂ / n mile]')
    top_5_least_efficient_str = format_dict_list("Top 5 Least Efficient Vessels (Highest CO2/nm)", insights.get('top_5_least_efficient', []), 'CO₂ emissions per distance [kg CO₂ / n mile]')

    pareto = insights.get('pareto_analysis', {})
    pareto_str = (f"The top 10% of vessels ({pareto.get('top_10_percent_count', 0)} ships) "
                  f"account for {pareto.get('top_10_percent_impact_pct', 0):.2f}% "
                  f"of the total fleet emissions ({pareto.get('total_fleet_emissions_kg', 0):,.0f} kg).")

    corr_activity = insights.get('correlation', {}).get('activity_vs_total_emissions', {})
    corr_activity_str = (f"- Time vs. Total Emissions: {corr_activity.get('time_vs_total_emissions', 0):.3f}\n"
                         f"- Distance vs. Total Emissions: {corr_activity.get('distance_vs_total_emissions', 0):.3f}")

    corr_efficiency = insights.get('correlation', {}).get('size_vs_efficiency', {})
    corr_efficiency_str = (f"- Length vs. Efficiency (CO2/nm): {corr_efficiency.get('length_vs_efficiency', 0):.3f}\n"
                           f"- Width vs. Efficiency (CO2/nm): {corr_efficiency.get('width_vs_efficiency', 0):.3f}\n"
                           f"- Draft vs. Efficiency (CO2/nm): {corr_efficiency.get('draft_vs_efficiency', 0):.3f}")

    prompt = f"""
You are a world-class shipping industry analyst and sustainability expert.
Your task is to write a concise, professional executive summary for a report on
fleet performance. Use the following key insights to write your summary.
Do not just list the data; interpret it. Focus on the *meaning* of the numbers.

--- KEY INSIGHTS ---

[Overall Fleet Statistics]
- Total Vessels Analyzed: {insights.get('total_vessels', 0)}
- Total Valid Efficiency Entries: {insights.get('valid_efficiency_entries', 0)} ({insights.get('invalid_entries', 0)} entries were invalid/missing)

[Fleet Efficiency (CO2/nm) Distribution]
- 75th Percentile: {insights.get('fleet_efficiency_stats', {}).get('75th_percentile', 0):,.2f} kg/nm
- 50th Percentile (Median): {insights.get('fleet_efficiency_stats', {}).get('median', 0):,.2f} kg/nm
- 25th Percentile: {insights.get('fleet_efficiency_stats', {}).get('25th_percentile', 0):,.2f} kg/nm
- Note: The mean ({insights.get('fleet_efficiency_stats', {}).get('mean', 0):,.2f} kg/nm) is heavily skewed by outliers and should be used with caution.

[Fleet Activity (Median)]
- Median Time at Sea: {insights.get('time_at_sea_stats', {}).get('50%', 0):,.2f} hours
- Median Distance Traveled: {insights.get('distance_stats', {}).get('50%', 0):,.2f} nautical miles

[Top Emitter Impact (Pareto Analysis)]
- {pareto_str}

{avg_by_shiptype_str}
{avg_by_rating_str}
{top_5_efficient_str}
{top_5_least_efficient_str}
[Correlation Analysis 1: Activity vs. Total Emissions]
{corr_activity_str}
- Interpretation: Analyze the strength of this logical correlation. A high value confirms that ships emitting more CO2 are those actively sailing.

[Correlation Analysis 2: Size vs. Efficiency]
{corr_efficiency_str}
- Interpretation: Analyze if larger ships (length/width/draft) are more or less efficient (CO2 per nautical mile).

--- END OF INSIGHTS ---

Please write the executive summary now. Start directly with the text.
"""
    return prompt.strip()

def call_watsonx_ai(prompt):
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        error_msg = "Watsonx API key or Project ID not found. Set env vars."
        print(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    print(f"--- Calling Watsonx AI (Model: {DEFAULT_MODEL_ID}) ---")

    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1024,
        "min_new_tokens": 50,
        "temperature": 0.2,
        "repetition_penalty": 1.05,
    }

    try:
        llm = WatsonxLLM(
            model_id=DEFAULT_MODEL_ID,
            url=WATSONX_URL,
            apikey=WATSONX_API_KEY,
            project_id=WATSONX_PROJECT_ID,
            params=parameters,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

        response = llm.invoke(prompt)

        print("--- Successfully received response from Watsonx AI ---")
        return response.strip()

    except requests.exceptions.ConnectionError:
        error_msg = f"Connection failed. Is watsonx fine? (Tried to connect to {WATSONX_URL})"
        print(f"ERROR: {error_msg}")
        raise ConnectionError(error_msg)

    except Exception as e:
        error_msg = f"Watsonx AI API call failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        raise RuntimeError(error_msg)


@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            insights = analyze_dataframe(df)

            prompt_for_llm = generate_prompt_from_insights(insights)

            elapsed = time.time() - start_time
            if elapsed < 2.0:
                time.sleep(2.0 - elapsed)

            return jsonify({
                "generated_prompt": prompt_for_llm,
                "key_insights": json.loads(json.dumps(insights, default=str))
            })

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type, please upload a .csv"}), 400

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    start_time = time.time()
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided in request."}), 400
        
    prompt = data['prompt']
    
    try:
        llm_response = call_watsonx_ai(prompt)
        
        elapsed = time.time() - start_time
        if elapsed < 2.0:
            time.sleep(2.0 - elapsed)

        return jsonify({
            "llm_response": llm_response
        })
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during generation: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)