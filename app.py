import pandas as pd
import numpy as np
import streamlit as st
from llama_cpp import Llama

st.set_page_config(page_title="Airtel vs Jio Competitive Dashboard", layout="wide")

# --- CONFIGURATION ---
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf"

@st.cache_resource
def load_llm():
    try:
        return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- STEP 1: LOAD & ENGINEER FEATURES (FULLY VECTORIZED & OPTIMIZED) ---
@st.cache_data
def load_and_prepare_data(file1, file2):
    df = pd.read_csv(file1)
    df['ASiteID 4G PL in GB'] = pd.to_numeric(df['ASiteID 4G PL in GB'], errors='coerce').fillna(0)
    df['ASiteID 5G PL in GB'] = pd.to_numeric(df['ASiteID 5G PL in GB'], errors='coerce').fillna(0)
    df['ASiteID Total PL in GB'] = df['ASiteID 4G PL in GB'] + df['ASiteID 5G PL in GB']
    
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    df2 = pd.read_csv(file2)
    df2['4G PL in GB'] = pd.to_numeric(df2['4G PL in GB'], errors='coerce').fillna(0)
    df2['5G PL in GB'] = pd.to_numeric(df2['5G PL in GB'], errors='coerce').fillna(0)
    df2['Total PL in GB'] = df2['4G PL in GB'] + df2['5G PL in GB']
    
    df2['Latitude'] = pd.to_numeric(df2['Latitude'], errors='coerce')
    df2['Longitude'] = pd.to_numeric(df2['Longitude'], errors='coerce')
    
    # --- CROSS-PROXIMITY LOGIC DEFINITIONS ---
    df['is_5g'] = df['ASiteID 5G'].astype(str).str.strip().str.lower() == 'yes'
    df2['is_5g'] = df2['5G Tag'].astype(str).str.strip().str.lower() == 'yes'
    
    a_5g_coords = df[df['is_5g'] & df['Latitude'].notnull() & df['Longitude'].notnull()]
    a_4g_800_coords = df[(~df['is_5g']) & (df['ASiteID Total PL in GB'] > 800) & df['Latitude'].notnull() & df['Longitude'].notnull()]
    
    j_5g_coords = df2[df2['is_5g'] & df2['Latitude'].notnull() & df2['Longitude'].notnull()]
    j_4g_800_coords = df2[(~df2['is_5g']) & (df2['Total PL in GB'] > 800) & df2['Latitude'].notnull() & df2['Longitude'].notnull()]
    
    # --- PERFORMANCE FIX: Chunked Vectorized Haversine Array Mathematics ---
    def get_nearby_mask(source_df, target_coords_df, threshold=500):
        if source_df.empty or target_coords_df.empty:
            return np.zeros(len(source_df), dtype=bool)
            
        lat2 = np.radians(target_coords_df['Latitude'].values)
        lon2 = np.radians(target_coords_df['Longitude'].values)
        
        lat1 = np.radians(source_df['Latitude'].values)
        lon1 = np.radians(source_df['Longitude'].values)
        
        mask = np.zeros(len(source_df), dtype=bool)
        chunk_size = 2000  # Process in memory-safe chunks
        
        for i in range(0, len(lat1), chunk_size):
            l1_chunk = lat1[i:i+chunk_size, np.newaxis]
            lo1_chunk = lon1[i:i+chunk_size, np.newaxis]
            
            dlat = lat2[np.newaxis, :] - l1_chunk
            dlon = lon2[np.newaxis, :] - lo1_chunk
            
            a = np.sin(dlat/2.0)**2 + np.cos(l1_chunk) * np.cos(lat2[np.newaxis, :]) * np.sin(dlon/2.0)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dist_meters = 6371000 * c
            
            mask[i:i+chunk_size] = np.any(dist_meters <= threshold, axis=1)
            
        return mask

    # Calculate boolean proximity flags globally instantly
    df['Near_J5'] = get_nearby_mask(df, j_5g_coords)
    df['Near_J4_800'] = get_nearby_mask(df, j_4g_800_coords)
    
    df2['Near_A5'] = get_nearby_mask(df2, a_5g_coords)
    df2['Near_A4_800'] = get_nearby_mask(df2, a_4g_800_coords)

    return df, df2

# --- STEP 2: AGGREGATE DATA ---
def aggregate_territory(territory_df, territory_df2):
    total_airtel = len(territory_df)
    total_jio = len(territory_df2)
    
    airtel_5g_df = territory_df[territory_df['is_5g']]
    jio_5g_df = territory_df2[territory_df2['is_5g']]
    
    # Execute strict SWOTN calculations per the provided Logic Table
    airtel_strength = territory_df[territory_df['is_5g'] & ~territory_df['Near_J5']].shape[0]
    airtel_weakness = territory_df[~territory_df['is_5g'] & territory_df['Near_J5']].shape[0]
    airtel_opportunity = territory_df[~territory_df['is_5g'] & ~territory_df['Near_J5'] & territory_df['Near_J4_800']].shape[0]
    airtel_neutral = territory_df[territory_df['is_5g'] & territory_df['Near_J5']].shape[0]
    
    # Evaluate Jio side to map Threat and calculate Jio's own table metrics
    jio_strength = territory_df2[territory_df2['is_5g'] & ~territory_df2['Near_A5']].shape[0]
    jio_weakness = territory_df2[~territory_df2['is_5g'] & territory_df2['Near_A5']].shape[0]
    jio_opportunity = territory_df2[~territory_df2['is_5g'] & ~territory_df2['Near_A5'] & territory_df2['Near_A4_800']].shape[0]
    jio_neutral = territory_df2[territory_df2['is_5g'] & territory_df2['Near_A5']].shape[0]
    
    airtel_threat = jio_opportunity 
    
    stats = {
        "Total_Airtel_Sites": total_airtel,
        "Total_Jio_Sites": total_jio,
        "Airtel_5G_Sites": airtel_5g_df.shape[0],
        "Jio_5G_Sites": jio_5g_df.shape[0],
        
        "Avg_Airtel_Total_PL": round(territory_df['ASiteID Total PL in GB'].mean(), 2) if not territory_df.empty else 0,
        "Avg_Jio_Total_PL": round(territory_df2['Total PL in GB'].mean(), 2) if not territory_df2.empty else 0,
        
        "Avg_Airtel_5G_PL": round(airtel_5g_df['ASiteID 5G PL in GB'].mean(), 2) if not airtel_5g_df.empty else 0,
        "Avg_Jio_5G_PL": round(jio_5g_df['5G PL in GB'].mean(), 2) if not jio_5g_df.empty else 0,
        
        "Avg_Airtel_4G_PL": round(territory_df['ASiteID 4G PL in GB'].mean(), 2) if not territory_df.empty else 0,
        "Avg_Jio_4G_PL": round(territory_df2['4G PL in GB'].mean(), 2) if not territory_df2.empty else 0,
        
        "Airtel_Influenced_By_Jio_5G": territory_df['Near_J5'].sum() if not territory_df.empty else 0,
        "Jio_Influenced_By_Airtel_5G": territory_df2['Near_A5'].sum() if not territory_df2.empty else 0,
        
        "SWOT_S": airtel_strength,
        "SWOT_W": airtel_weakness,
        "SWOT_O": airtel_opportunity,
        "SWOT_T": airtel_threat,
        "SWOT_N": airtel_neutral,
        
        # Jio Specific Table Metrics
        "Jio_S": jio_strength,
        "Jio_W": jio_weakness,
        "Jio_O": jio_opportunity,
        "Jio_T": airtel_opportunity, # Inverse mapped per logic table
        "Jio_N": jio_neutral
    }
    
    stats["Airtel_5G_Percent"] = round((stats["Airtel_5G_Sites"] / total_airtel) * 100, 2) if total_airtel > 0 else 0
    stats["Jio_5G_Percent"] = round((stats["Jio_5G_Sites"] / total_jio) * 100, 2) if total_jio > 0 else 0
    return stats

# --- STEP 3: OFFLINE LLM INTEGRATION (STRICT FORMATTING) ---
def generate_swot_offline(location_name, stats, active_llm):
    if not active_llm:
        return "Model not loaded. Please check your MODEL_PATH or system memory."

    facts = []
    if stats['SWOT_S'] > 0:
        facts.append(f"STRENGTH: Airtel has {stats['SWOT_S']} exclusive 5G sites with absolutely no Jio 5G presence within 500 meters.")
    if stats['SWOT_W'] > 0:
        facts.append(f"WEAKNESS: Airtel operates {stats['SWOT_W']} 4G sites that are threatened by the presence of a Jio 5G site within 500 meters.")
    if stats['SWOT_O'] > 0:
        facts.append(f"OPPORTUNITY: Airtel has {stats['SWOT_O']} 4G sites (with no Jio 5G nearby) situated near heavily loaded Jio 4G sites (>800 GB payload), presenting prime upgrade targets.")
    if stats['SWOT_T'] > 0:
        facts.append(f"THREAT: Jio operates {stats['SWOT_T']} 4G sites situated near heavily loaded Airtel 4G sites (>800 GB payload), putting high-volume Airtel traffic at risk of churn.")
    if stats['SWOT_N'] > 0:
        facts.append(f"NEUTRAL: Both operators maintain a competitive stalemate at {stats['SWOT_N']} locations where both Airtel and Jio have active 5G sites within 500 meters.")

    facts_text = "\n- ".join(facts) if facts else "No definitive SWOTN data points available based on strict geographic proximity rules."

    prompt = f"""[INST] You are an Executive Presentation formatter. Format the following pre-calculated, verified SWOTN facts for AIRTEL in {location_name}.
    
    FACTS TO FORMAT:
    - {facts_text}

    Task: Create a professional, bulleted SWOTN analysis (Strengths, Weaknesses, Opportunities, Threats, Neutral). 
    CRITICAL RULE: DO NOT invent, calculate, or add any new logical conclusions. Just present the exact facts provided above in a readable, executive format. [/INST]"""
    
    try:
        response = active_llm(
            prompt,
            max_tokens=350,
            temperature=0.1,  
            stop=["</s>"]
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"Error generating text: {e}"

# --- STEP 4: DASHBOARD UI ---
def main():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1.5rem;
                    padding-bottom: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
        
    st.title("📡 Airtel vs Jio: Competitive Intelligence Dashboard")
    
    try:
        df, df2 = load_and_prepare_data("input.csv", "input2.csv")
    except FileNotFoundError:
        st.error("Please place both 'input.csv' and 'input2.csv' in the same directory.")
        return

    st.sidebar.header("Territory Filters")
    
    # --- FIX: Case-Insensitive Matching for Airtel and Jio Geographies ---
    
    # 1. State Filter
    state_opts = [s.title() for s in df['State'].dropna().astype(str).str.lower().unique()]
    state = st.sidebar.selectbox("Select State", state_opts)
    state_df = df[df['State'].astype(str).str.lower() == str(state).lower()]
    
    # Align Jio data using case-insensitive district matching
    valid_districts_lower = state_df['District Name'].dropna().astype(str).str.lower().unique()
    state_df2 = df2[df2['District'].astype(str).str.lower().isin(valid_districts_lower)]
    
    # 2. District Filter
    district_opts = ["All"] + [d.title() for d in valid_districts_lower]
    district = st.sidebar.selectbox("Select District", district_opts)
    
    dist_df = state_df if district == "All" else state_df[state_df['District Name'].astype(str).str.lower() == str(district).lower()]
    dist_df2 = state_df2 if district == "All" else state_df2[state_df2['District'].astype(str).str.lower() == str(district).lower()]
    
    # 3. Tehsil Filter
    valid_tehsils_lower = dist_df['Tehsil Name'].dropna().astype(str).str.lower().unique()
    tehsil_opts = ["All"] + [t.title() for t in valid_tehsils_lower]
    tehsil = st.sidebar.selectbox("Select Tehsil", tehsil_opts)
    
    tehsil_df = dist_df if tehsil == "All" else dist_df[dist_df['Tehsil Name'].astype(str).str.lower() == str(tehsil).lower()]
    tehsil_df2 = dist_df2 if tehsil == "All" else dist_df2[dist_df2['Tehsil'].astype(str).str.lower() == str(tehsil).lower()]
    
    # 4. Town Filter
    valid_towns_lower = tehsil_df['Town Name'].dropna().astype(str).str.lower().unique()
    town_opts = ["All"] + [t.title() for t in valid_towns_lower]
    town = st.sidebar.selectbox("Select Town", town_opts)
    
    filtered_df = tehsil_df if town == "All" else tehsil_df[tehsil_df['Town Name'].astype(str).str.lower() == str(town).lower()]
    filtered_df2 = tehsil_df2 if town == "All" else tehsil_df2[tehsil_df2['Town Name'].astype(str).str.lower() == str(town).lower()]
    
    location_label = town if town != "All" else (tehsil if tehsil != "All" else (district if district != "All" else state))

    stats = aggregate_territory(filtered_df, filtered_df2)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1.2, 1.2, 1.6])
    with col1:
        st.metric("Airtel Avg Total PL", f"{stats['Avg_Airtel_Total_PL']} GB")
        st.caption(f"**4G:** {stats['Avg_Airtel_4G_PL']} GB | **5G:** {stats['Avg_Airtel_5G_PL']} GB")
    with col2:
        st.metric("Jio Avg Total PL", f"{stats['Avg_Jio_Total_PL']} GB")
        st.caption(f"**4G:** {stats['Avg_Jio_4G_PL']} GB | **5G:** {stats['Avg_Jio_5G_PL']} GB")
    with col3:
        st.metric("Airtel 5G Sites", f"{stats['Airtel_5G_Sites']} ({stats['Airtel_5G_Percent']}%)")
    with col4:
        st.metric("Jio 5G Sites", f"{stats['Jio_5G_Sites']} ({stats['Jio_5G_Percent']}%)")
    with col5:
        st.metric("Influenced by Rival 5G (<500m)", f"Airtel: {stats['Airtel_Influenced_By_Jio_5G']} | Jio: {stats['Jio_Influenced_By_Airtel_5G']}")

    st.markdown("---")

    st.subheader("📊 SWOTN Analysis (Site Counts)")
    swotn_data = {
        "Category": ["Strength", "Weakness", "Opportunity", "Threat", "Neutral"],
        "Airtel": [stats['SWOT_S'], stats['SWOT_W'], stats['SWOT_O'], stats['SWOT_T'], stats['SWOT_N']],
        "Jio": [stats['Jio_S'], stats['Jio_W'], stats['Jio_O'], stats['Jio_T'], stats['Jio_N']]
    }
    st.dataframe(pd.DataFrame(swotn_data), width='stretch', hide_index=True)

    st.markdown("---")

    col_map, col_swot = st.columns([1.5, 1])

    with col_map:
        st.subheader(f"Geospatial View: {location_label}")
        
        map_frames = []
        
        if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
            map_data1 = filtered_df[['Latitude', 'Longitude']].copy()
            map_data1.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
            map_data1['color'] = '#FF0000'
            if not map_data1.empty:
                map_frames.append(map_data1)
            
        if 'Latitude' in filtered_df2.columns and 'Longitude' in filtered_df2.columns:
            map_data2 = filtered_df2[['Latitude', 'Longitude']].copy()
            map_data2.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
            map_data2['color'] = '#0000FF'
            if not map_data2.empty:
                map_frames.append(map_data2)
            
        if map_frames:
            map_coords = pd.concat(map_frames, ignore_index=True)
            map_coords['lat'] = pd.to_numeric(map_coords['lat'], errors='coerce')
            map_coords['lon'] = pd.to_numeric(map_coords['lon'], errors='coerce')
            st.map(map_coords.dropna(subset=['lat', 'lon']), color='color')
        else:
            st.warning("Latitude and Longitude columns missing or datasets are empty.")

    with col_swot:
        st.subheader("🤖 Local LLM SWOT Analysis")
        if st.button("Generate SWOT Insights"):
            with st.spinner("Loading Model & Analyzing locally..."):
                active_llm = load_llm()
                swot_text = generate_swot_offline(location_label, stats, active_llm)
                st.write(swot_text)
        else:
            st.info("Click the button to generate an AI analysis.")

if __name__ == "__main__":
    main()