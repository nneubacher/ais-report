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
WATSONX_URL = "https://eu-gb.ml.cloud.ibm.com"

DEFAULT_MODEL_ID = "ibm/granite-3-2b-instruct"

if not WATSONX_API_KEY:
    print("ERROR: WATSONX_API_KEY not set.")
if not WATSONX_PROJECT_ID:
    print("ERROR: WATSONX_PROJECT_ID not set.")

app = Flask(__name__)
CORS(app)

def clean_dataframe(df):
    """
    Takes a raw dataframe and returns a cleaned dataframe.
    - Converts all numeric columns.
    - Engineers 'Efficiency_Rating_Type' feature.
    """
    print("\n--- [Data Cleaner] Start ---")
    
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
            print(f"Warning: Expected numeric column '{col}' not found in CSV.")
    
    print("\n--- [Data Cleaner] After Numeric Conversion: ---")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())

    if 'Technical efficiency' in df.columns:
        df['Efficiency_Rating_Type'] = df['Technical efficiency'].astype(str).str.split(' ').str[0].str.strip('()')
        print("\n--- [Data Cleaner] Technical Rating Types Found: ---")
        print(df['Efficiency_Rating_Type'].value_counts())
    else:
        print("Warning: 'Technical efficiency' column not found.")
        df['Efficiency_Rating_Type'] = 'Unknown'
        
    return df

def analyze_fleet_overview(df):
    """
    Performs the original "Fleet Overview" analysis.
    """
    insights = {}
    print("\n--- [Insight Engine] Running: General Overview ---")
    
    valid_df = df[df['CO₂ emissions per distance [kg CO₂ / n mile]'] > 0].copy()
    
    insights['total_vessels'] = len(df)
    insights['valid_efficiency_entries'] = len(valid_df)
    insights['invalid_entries'] = insights['total_vessels'] - insights['valid_efficiency_entries']

    # Fleet-wide Statistics
    insights['fleet_efficiency_stats'] = valid_df['CO₂ emissions per distance [kg CO₂ / n mile]'].describe().to_dict()
    insights['time_at_sea_stats'] = df['Time spent at sea [hours]'].describe().to_dict()
    insights['distance_stats'] = df['total_distance_nm'].describe().to_dict()

    # Analysis by Ship Type
    insights['avg_by_shiptype'] = valid_df.groupby('Ship type')['CO₂ emissions per distance [kg CO₂ / n mile]'].mean().sort_values(ascending=False).to_dict()

    # Analysis by Technical Rating
    avg_by_rating = valid_df.groupby('Efficiency_Rating_Type')['CO₂ emissions per distance [kg CO₂ / n mile]'].mean().sort_values(ascending=False)
    if not avg_by_rating.empty:
        insights['avg_by_rating'] = avg_by_rating.to_dict()

    # Top/Bottom Performers
    insights['top_5_efficient'] = valid_df.sort_values(by='CO₂ emissions per distance [kg CO₂ / n mile]', ascending=True).head(5)[['Name', 'Ship type', 'CO₂ emissions per distance [kg CO₂ / n mile]']].to_dict('records')
    
    least_efficient_df = df.copy()
    least_efficient_df['CO₂ emissions per distance [kg CO₂ / n mile]'] = least_efficient_df['CO₂ emissions per distance [kg CO₂ / n mile]'].fillna(0)
    insights['top_5_least_efficient'] = least_efficient_df.sort_values(by='CO₂ emissions per distance [kg CO₂ / n mile]', ascending=False).head(5)[['Name', 'Ship type', 'CO₂ emissions per distance [kg CO₂ / n mile]']].to_dict('records')

    # Top Emitter Impact (Pareto Principle)
    df_sorted_by_total = df.sort_values(by='total_CO₂_emissions_kg_CO₂', ascending=False)
    total_fleet_emissions = df_sorted_by_total['total_CO₂_emissions_kg_CO₂'].sum()
    
    num_top_10_percent = int(len(df_sorted_by_total) * 0.10)
    top_10_percent_emissions = df_sorted_by_total.head(num_top_10_percent)['total_CO₂_emissions_kg_CO₂'].sum()
    
    insights['pareto_analysis'] = {
        'top_10_percent_count': num_top_10_percent,
        'top_10_percent_impact_pct': (top_10_percent_emissions / total_fleet_emissions * 100) if total_fleet_emissions > 0 else 0,
        'total_fleet_emissions_kg': total_fleet_emissions
    }

    # Correlation Analysis
    insights['correlation'] = {}
    corr_cols_activity = ['Time spent at sea [hours]', 'total_distance_nm', 'total_CO₂_emissions_kg_CO₂']
    if all(col in df.columns for col in corr_cols_activity):
        corr_matrix_activity = df[corr_cols_activity].dropna().corr()
        if not corr_matrix_activity.empty:
            insights['correlation']['activity_vs_total_emissions'] = {
                'time_vs_total_emissions': corr_matrix_activity.loc['Time spent at sea [hours]', 'total_CO₂_emissions_kg_CO₂'],
                'distance_vs_total_emissions': corr_matrix_activity.loc['total_distance_nm', 'total_CO₂_emissions_kg_CO₂']
            }
    
    corr_cols_efficiency = ['Length', 'Width', 'Draft', 'CO₂ emissions per distance [kg CO₂ / n mile]']
    if all(col in valid_df.columns for col in corr_cols_efficiency):
        corr_matrix_efficiency = valid_df[corr_cols_efficiency].dropna().corr()
        if not corr_matrix_efficiency.empty:
            insights['correlation']['size_vs_efficiency'] = {
                'length_vs_efficiency': corr_matrix_efficiency.loc['Length', 'CO₂ emissions per distance [kg CO₂ / n mile]'],
                'width_vs_efficiency': corr_matrix_efficiency.loc['Width', 'CO₂ emissions per distance [kg CO₂ / n mile]'],
                'draft_vs_efficiency': corr_matrix_efficiency.loc['Draft', 'CO₂ emissions per distance [kg CO₂ / n mile]']
            }
            
    print("\n--- [Insight Engine] General Overview Complete ---")
    return insights

def generate_fleet_overview_prompt(insights):
    """
    Generates the prompt for the "Fleet Overview" analysis.
    (This is the original prompt generator, renamed)
    """
    
    # Helper to format list of dicts into a clean string
    def format_dict_list(title, data_list, value_key):
        s = f"[{title}]\n"
        if not data_list:
            return f"[{title}]\n- (No data available)\n"
        for i, item in enumerate(data_list, 1):
            s += f"{i}. {item['Name']} ({item['Ship type']}): {item.get(value_key, 0):,.2f} kg/nm\n"
        return s

    # Helper to format the ship type/rating averages
    def format_dict_averages(title, data_dict):
        s = f"[{title}]\n"
        if not data_dict:
            return f"[{title}]\n- (No data available)\n"
        for key, avg in data_dict.items():
            s += f"- {key}: {avg:,.2f} kg/nm\n"
        return s

    # Build prompt sections
    avg_by_shiptype_str = format_dict_averages("Efficiency by Ship Type (Avg CO2/nm)", insights.get('avg_by_shiptype', {}))
    avg_by_rating_str = format_dict_averages("Efficiency by Technical Rating (Avg CO2/nm)", insights.get('avg_by_rating', {}))
    top_5_efficient_str = format_dict_list("Top 5 Most Efficient Vessels (Lowest CO2/nm)", insights.get('top_5_efficient', []), 'CO₂ emissions per distance [kg CO₂ / n mile]')
    top_5_least_efficient_str = format_dict_list("Top 5 Least Efficient Vessels (Highest CO2/nm)", insights.get('top_5_least_efficient', []), 'CO₂ emissions per distance [kg CO₂ / n mile]')
    
    pareto = insights.get('pareto_analysis', {})
    pareto_str = (f"The top 10% of vessels ({pareto.get('top_10_percent_count', 0)} ships) "
                  f"account for {pareto.get('top_10_percent_impact_pct', 0):.2f}% "
                  f"of the total emissions ({pareto.get('total_fleet_emissions_kg', 0):,.0f} kg).")

    corr_activity = insights.get('correlation', {}).get('activity_vs_total_emissions', {})
    corr_activity_str = (f"- Time vs. Total Emissions: {corr_activity.get('time_vs_total_emissions', 0):.3f}\n"
                         f"- Distance vs. Total Emissions: {corr_activity.get('distance_vs_total_emissions', 0):.3f}")
    
    corr_efficiency = insights.get('correlation', {}).get('size_vs_efficiency', {})
    corr_efficiency_str = (f"- Length vs. Efficiency (CO2/nm): {corr_efficiency.get('length_vs_efficiency', 0):.3f}\n"
                           f"- Width vs. Efficiency (CO2/nm): {corr_efficiency.get('width_vs_efficiency', 0):.3f}\n"
                           f"- Draft vs. Efficiency (CO2/nm): {corr_efficiency.get('draft_vs_efficiency', 0):.3f}")

    # The Final Prompt
    prompt = f"""
You are a world-class shipping industry analyst and sustainability expert.
Your task is to write a concise, professional executive summary for a report on
general performance. Use the following key insights to write your summary.
Do not just list the data; interpret it. Focus on the *meaning* of the numbers.

--- KEY INSIGHTS (GENERAL OVERVIEW) ---

[Overall Statistics]
- Total Vessels Analyzed: {insights.get('total_vessels', 0)}
- Total Valid Efficiency Entries: {insights.get('valid_efficiency_entries', 0)} ({insights.get('invalid_entries', 0)} entries were invalid/missing)

[General Efficiency (CO2/nm) Distribution]
- 75th Percentile: {insights.get('fleet_efficiency_stats', {}).get('75%', 0):,.2f} kg/nm
- 50th Percentile (Median): {insights.get('fleet_efficiency_stats', {}).get('50%', 0):,.2f} kg/nm
- 25th Percentile: {insights.get('fleet_efficiency_stats', {}).get('25%', 0):,.2f} kg/nm
- Note: The mean ({insights.get('fleet_efficiency_stats', {}).get('mean', 0):,.2f} kg/nm) is heavily skewed by outliers and should be used with caution.

[General Activity (Median)]
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

Please write the executive summary now. Start by presenting the key insights in a structured manner and after summarize the findings in human-language.
"""
    return prompt.strip()


# --- NEW: Step 2b - Company Deep Dive Analysis ---
def analyze_company_deep_dive(df, company_name):
    """
    Performs the new "Company Deep Dive" analysis.
    Filters for the company and compares its stats to the general average.
    """
    insights = {}
    print(f"\n--- [Insight Engine] Running: Company Deep Dive for: {company_name} ---")
    
    # --- 1. Filter for the company ---
    # Use str.contains for partial, case-insensitive matching
    company_df = df[df['Name_1'].str.contains(company_name, case=False, na=False)].copy()
    
    if company_df.empty:
        raise ValueError(f"No company found matching the name '{company_name}' in the 'Name_1' column.")
        
    insights['company_name'] = company_name
    insights['vessel_count'] = len(company_df)
    
    # --- 2. Get Company Stats ---
    company_valid_df = company_df[company_df['CO₂ emissions per distance [kg CO₂ / n mile]'] > 0]
    
    insights['company_stats'] = {
        'median_efficiency_kg_nm': company_valid_df['CO₂ emissions per distance [kg CO₂ / n mile]'].median(),
        'median_time_at_sea_hr': company_df['Time spent at sea [hours]'].median(),
        'median_distance_nm': company_df['total_distance_nm'].median(),
        'total_emissions_kg': company_df['total_CO₂_emissions_kg_CO₂'].sum(),
        'vessel_list': company_df['Name'].tolist()
    }
    
    # --- 3. Get Fleet-wide Stats for Comparison ---
    fleet_valid_df = df[df['CO₂ emissions per distance [kg CO₂ / n mile]'] > 0]
    
    insights['fleet_comparison_stats'] = {
        'median_efficiency_kg_nm': fleet_valid_df['CO₂ emissions per distance [kg CO₂ / n mile]'].median(),
        'median_time_at_sea_hr': df['Time spent at sea [hours]'].median(),
        'median_distance_nm': df['total_distance_nm'].median(),
        'average_vessel_emissions_kg': df['total_CO₂_emissions_kg_CO₂'].mean()
    }
    
    print("\n--- [Insight Engine] Company Deep Dive Complete ---")
    return insights

# --- NEW: Step 3b - Company Deep Dive Prompt ---
def generate_company_deep_dive_prompt(insights):
    """
    Generates the new prompt for the "Company Deep Dive" analysis.
    """
    
    company = insights.get('company_name', 'The Company')
    company_stats = insights.get('company_stats', {})
    fleet_stats = insights.get('fleet_comparison_stats', {})
    
    prompt = f"""
You are a professional data auditor and sustainability analyst.
Your task is to write an executive summary comparing a specific company's fleet performance against the overall average of all tracked companies.
Focus on whether the company is performing better or worse than the average.

--- KEY INSIGHTS (COMPANY DEEP DIVE) ---

[Company Analyzed]
- Name: {company}
- Vessels Found: {insights.get('vessel_count', 0)}
- Vessel Names: {', '.join(company_stats.get('vessel_list', []))}

[Company Performance]
- Median Efficiency: {company_stats.get('median_efficiency_kg_nm', 0):,.2f} kg CO2/nm
- Median Time at Sea: {company_stats.get('median_time_at_sea_hr', 0):,.2f} hours
- Median Distance Traveled: {company_stats.get('median_distance_nm', 0):,.2f} nm
- Total Company Fleet Emissions: {company_stats.get('total_emissions_kg', 0):,.0f} kg CO2

[General Average (for Comparison)]
- General Median Efficiency: {fleet_stats.get('median_efficiency_kg_nm', 0):,.2f} kg CO2/nm
- General Median Time at Sea: {fleet_stats.get('median_time_at_sea_hr', 0):,.2f} hours
- General Median Distance Traveled: {fleet_stats.get('median_distance_nm', 0):,.2f} nm
- General Avg. Total Emissions (per vessel): {fleet_stats.get('average_vessel_emissions_kg', 0):,.0f} kg CO2

--- END OF INSIGHTS ---

Please write the executive summary now. Start by presenting the key insights in a structured manner and after summarize the findings in human-language.
Start by stating which company you are analyzing.
Compare the company's efficiency, activity, and total emissions to all tracked companies,
and conclude whether they are a high or low performer.
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

@app.route('/get-company-list', methods=['POST'])
def get_company_list_endpoint():
    """
    This new endpoint reads the uploaded CSV and returns a
    unique, sorted list of company names from the 'Name_1' column.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            if 'Name_1' not in df.columns:
                return jsonify({"error": "Column 'Name_1' not found in CSV."}), 400
                
            # Get unique, non-null, non-empty names
            names = df['Name_1'].dropna().unique().tolist()
            clean_names = [name for name in names if isinstance(name, str) and name.strip()]
            clean_names.sort() # Sort them alphabetically
            
            return jsonify({"companies": clean_names})

        except Exception as e:
            return jsonify({"error": f"An error occurred while reading companies: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type, please upload a .csv"}), 400

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    start_time = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Get analysis type and company name from form
    analysis_type = request.form.get('analysis_type', 'fleet_overview')
    company_name = request.form.get('company_name', None)

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            cleaned_df = clean_dataframe(df)

            if analysis_type == 'company_deep_dive':
                if not company_name or company_name == "":
                    raise ValueError("Company name is required and was not provided for deep dive analysis.")
                insights = analyze_company_deep_dive(cleaned_df, company_name)
                prompt_for_llm = generate_company_deep_dive_prompt(insights)

            else: # Default to fleet_overview
                insights = analyze_fleet_overview(cleaned_df)
                prompt_for_llm = generate_fleet_overview_prompt(insights)

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