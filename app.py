import streamlit as st
import time
from PIL import Image
from optimizer_engine import PromptOptimizer
from sd_interface import StableDiffusionClient
from evaluator import PromptEvaluator
import pandas as pd

# Page Config
st.set_page_config(page_title="Prompt Optimizer & Enhancer", page_icon="✨", layout="wide")

# Initialize Sessions
if "optimizer" not in st.session_state:
    st.session_state.optimizer = PromptOptimizer()
if "sd_client" not in st.session_state:
    st.session_state.sd_client = StableDiffusionClient()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = PromptEvaluator()
if "results" not in st.session_state:
    st.session_state.results = {}
if "history" not in st.session_state:
    st.session_state.history = []

# Header
st.title("✨ NLP Prompt Optimizer & Enhancer")
st.markdown("""
This project transforms basic user prompts into **high-quality descriptive prompts** using a classical NLP pipeline. 
It then compares the output of **Stable Diffusion** for both the raw and optimized prompts.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")
sd_url = st.sidebar.text_input("Stable Diffusion URL", value="http://127.0.0.1:7860")
st.sidebar.markdown("---")
st.sidebar.info("Tokenize → SVO Map → TF-IDF → NP Chunk → Specificity → Synonym Swap → Genetic Evolution")
st.sidebar.markdown("---")
st.sidebar.subheader("📜 Optimization History")
if st.session_state.history:
    for idx, item in enumerate(reversed(st.session_state.history[-5:])):
        with st.sidebar.expander(f"Run #{len(st.session_state.history) - idx}"):
            st.write(f"**Original:** {item['original'][:30]}...")
            if st.button(f"Load Run #{len(st.session_state.history) - idx}", key=f"load_{idx}"):
                st.session_state.results = item
                st.rerun()
else:
    st.sidebar.write("No history yet.")
st.sidebar.subheader("Stage 2: Image Gen")
st.sidebar.info("Dual Path: Raw vs Optimized")
st.sidebar.subheader("Stage 3: Evaluation")
st.sidebar.info("CLIP Score + Token Density + Latency")

# Main Input
user_input = st.text_input("Enter your basic prompt:", placeholder="e.g., a girl eating food")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Optimize Prompt", use_container_width=True):
        if user_input:
            with st.spinner("Analyzing and optimizing..."):
                opt_result = st.session_state.optimizer.optimize(user_input)
                
                # Full State Update
                st.session_state.results = {
                    "original": user_input,
                    "optimized": opt_result['optimized_prompt'],
                    "pipeline_log": opt_result.get('pipeline_log', []),
                    "pipeline_stages": opt_result.get('pipeline_stages', []),
                    "linguistics": opt_result.get('linguistics', []),
                    "sd_settings": opt_result['settings'],
                    "fitness_score": opt_result.get('fitness_score', 0.0),
                    "noun_phrases": opt_result.get('noun_phrases', []),
                    "entities": opt_result.get('entities', ""),
                    "svo_triplets": opt_result.get('svo_triplets', [])
                }
                
                # Update Session History
                st.session_state.history.append(st.session_state.results.copy())
                
            st.success("NLP Optimization Complete!")
            st.code(f"Original: {user_input}\nOptimized: {opt_result['optimized_prompt']}", language="markdown")
            
            with st.expander("🔍 See NLP Pipeline Steps"):
                st.write("### 🏷️ Part-of-Speech & Keywords")
                ling_df = pd.DataFrame(opt_result.get('linguistics', []))
                if not ling_df.empty:
                    # Specificity visualization
                    st.dataframe(ling_df[['word', 'pos', 'label', 'tfidf_score', 'optimized_to']], use_container_width=True)
                    
                    st.write("### 🪜 Wordnet Abstraction Ladder (Specificity)")
                    for l in opt_result.get('linguistics', []):
                        if l.get('specificity'):
                            with st.expander(f"Ladder for: {l['word']}"):
                                ladder = l['specificity']['ladder']
                                st.write(" → ".join([f"**{w.upper()}**" if i == len(ladder)-1 else w for i, w in enumerate(ladder)]))
                                st.progress(min(l['specificity']['depth'] / 10, 1.0))
                                st.caption(f"Depth: {l['specificity']['depth']} | {'✅ Specific' if not l['specificity']['is_generic'] else '⚠️ Generic'}")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("### 🧩 Noun Phrases")
                    for np in opt_result.get('noun_phrases', []):
                        st.info(np)
                    st.write("### 🔄 SVO Triplets (Semantic Action)")
                    for svo in opt_result.get('svo_triplets', []):
                        st.code(f"{svo['subject']} --[{svo['action']}]--> {svo['object']}")
                with col_b:
                    st.write("### 🤖 Named Entities")
                    if opt_result.get('entities'):
                        st.success(opt_result.get('entities'))
                    else:
                        st.write("None detected.")

                st.write("### 📜 Process Log")
                for line in opt_result['pipeline_log']:
                    st.write(f"- {line}")
        else:
            st.warning("Please enter a prompt first.")

with col2:
    if st.button("🖼 Generate & Compare Images", use_container_width=True):
        if 'optimized' in st.session_state.results:
            results = st.session_state.results
            with st.spinner("Generating dual images (this takes ~30-60s)..."):
                # Raw Prompt Gen
                raw_gen = st.session_state.sd_client.generate_image(results['original'])
                # Optimized Prompt Gen with suggested settings
                opt_gen = st.session_state.sd_client.generate_image(
                    results['optimized'], 
                    steps=results['sd_settings']['steps'],
                    cfg_scale=results['sd_settings']['cfg_scale']
                )
                
                if raw_gen['status'] == 'success' and opt_gen['status'] == 'success':
                    # Evaluation
                    raw_clip = st.session_state.evaluator.calculate_clip_score(raw_gen['image'], results['original'])
                    opt_clip = st.session_state.evaluator.calculate_clip_score(opt_gen['image'], results['optimized'])
                    
                    st.session_state.results['raw_img'] = raw_gen['image']
                    st.session_state.results['opt_img'] = opt_gen['image']
                    st.session_state.results['raw_latency'] = raw_gen['inference_time']
                    st.session_state.results['opt_latency'] = opt_gen['inference_time']
                    st.session_state.results['raw_clip'] = raw_clip
                    st.session_state.results['opt_clip'] = opt_clip
                    
                    # New STS Metric
                    sts_score = st.session_state.evaluator.calculate_sts_score(results['original'], results['optimized'])
                    st.session_state.results['sts_score'] = sts_score
                    
                    st.session_state.results['raw_tokens'] = st.session_state.evaluator.get_token_count(results['original'])
                    st.session_state.results['opt_tokens'] = st.session_state.evaluator.get_token_count(results['optimized'])
                else:
                    st.error("Failed to connect to Stable Diffusion. Is the --api flag enabled?")
        else:
            st.warning("Optimize the prompt first!")

# Comparison Section
if 'raw_img' in st.session_state.results:
    st.divider()
    res = st.session_state.results
    
    st.subheader("📊 Comparison Dashboard")
    
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.write("### Prompt A (Raw)")
        st.write(f"*{res['original']}*")
        st.image(res['raw_img'], use_container_width=True, caption="Generated from raw input")
        
    with img_col2:
        st.write("### Prompt B (Optimized)")
        st.write(f"*{res['optimized']}*")
        st.image(res['opt_img'], use_container_width=True, caption="Generated from NLP-enhanced input")

    # Metrics Table
    st.subheader("🧠 Performance Metrics")
    metrics_data = {
        "Metric": ["CLIP Score (Similarity)", "STS Score (Meaning Preservation)", "Token Count", "Inference Time (s)", "GA Fitness Score"],
        "Raw Prompt": [res['raw_clip'], "1.000 (Baseline)", res['raw_tokens'], f"{res['raw_latency']:.2f}", "N/A"],
        "Optimized Prompt": [res['opt_clip'], res.get('sts_score', 0.0), res['opt_tokens'], f"{res['opt_latency']:.2f}", f"{res.get('fitness_score', 0.0):.2f}"]
    }
    df = pd.DataFrame(metrics_data)
    st.table(df)
    
    # Comparison Summary
    clip_diff = res['opt_clip'] - res['raw_clip']
    if clip_diff > 0:
        st.success(f"🚀 **Improvement Detected!** The optimized prompt achieved a **{clip_diff:.4f}** higher CLIP score than the raw input.")
    else:
        st.info("The scores are similar, but the visual complexity is noticeably higher in the optimized version.")

# Pipeline Visualization (Static)
st.divider()
with st.expander("🧱 View Project Pipeline Diagram"):
    st.image("https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui-assets/master/screenshots/txt2img.png", width=300) # Placeholder or actual diagram
    st.markdown("""
    **Step 1: User Input** → Raw query
    **Step 2: Preprocessing** → Cleaning & Tokenization
    **Step 3: NLP Module** → **POS Tagging + Synonym Selection + Adjective Injection**
    **Step 4: Dual Path** → Compare original vs enhanced
    **Step 5: Image Generation** → Stable Diffusion API
    **Step 6: Evaluation** → CLIP Similarity + Performance Benchmarks
    """)
