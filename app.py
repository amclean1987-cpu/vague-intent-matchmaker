import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import os
from dotenv import load_dotenv

# 1. Setup & Config
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embed_model = "models/text-embedding-004"
text_model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Vague-Intent Matchmaker", layout="centered")
st.title("✈️ Vague-Intent Matchmaker")
st.write("Type your messy holiday brain-dump below. We'll find the match.")

# 2. Load Data & Embeddings
@st.cache_data
def load_and_embed_data():
    df = pd.read_csv("destinations.csv")
    # Generate embeddings for each destination's vibe
    embeddings = []
    for vibe in df['Vibe_Description']:
        emb = genai.embed_content(model=embed_model, content=vibe)['embedding']
        embeddings.append(emb)
    df['Embedding'] = embeddings
    return df

try:
    df = load_and_embed_data()
except Exception as e:
    st.error(f"Error loading destinations.csv: {e}\nMake sure the file is in the same folder!")
    st.stop()

def get_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 3. UI Input
user_query = st.text_area("What are you looking for?", placeholder="e.g., Cheap lads weekend, under 3 hours, need good nightlife...")

if st.button("Find Matches") and user_query:
    with st.spinner("Decoding intent and matching..."):
        # Embed user query
        query_emb = genai.embed_content(model=embed_model, content=user_query)['embedding']
        
        # Calculate similarities against the database
        df['Similarity'] = df['Embedding'].apply(lambda x: get_similarity(query_emb, x))
        
        # Pull the top 3 results
        top_3 = df.nlargest(3, 'Similarity')
        
        st.subheader("Your Top 3 Matches")
        
        # 4. Generate Pitches & Display
        for _, row in top_3.iterrows():
            prompt = f"""
            The user wants: '{user_query}'
            You are pitching: {row['Destination']} (Flight: {row['Flight_Time_From_UK']}h, Cost: {row['Estimated_Cost']})
            Vibe: {row['Vibe_Description']}
            
            Write a punchy, commercial 2-sentence pitch explaining exactly why this destination fits their request. No fluff.
            """
            pitch = text_model.generate_content(prompt).text
            
            with st.container():
                st.markdown(f"### {row['Destination']} | {row['Estimated_Cost']}")
                st.write(f"⏱️ **Flight:** {row['Flight_Time_From_UK']} hours from UK")
                st.write(pitch)
                st.divider()
