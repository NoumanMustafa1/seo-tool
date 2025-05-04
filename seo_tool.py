import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
import re
import time
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
import string

# Download necessary NLTK data
try:
    # Force download these resources at the beginning
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")
    # Fallback simple tokenizer function in case NLTK fails
    def word_tokenize(text):
        # Simple word tokenization as fallback
        return text.split()
    
    # Empty set as fallback for stopwords
    stopwords = type('', (), {'words': lambda x: []})
    stopwords.words = lambda x: []

# Configure page
st.set_page_config(
    page_title="Advanced SEO Keyword Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 5px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key input with secure handling
    api_key = st.text_input(
        "SerpAPI Key", 
        value="2f257a6ce3c6f7fd62fa27f0f426de494e8790f5a0ceb0a12d4c5f45b5008511",
        type="password",
        help="Enter your SerpAPI key. Get one at https://serpapi.com/"
    )
    
    st.markdown("---")
    
    # Advanced settings
    st.markdown("### Advanced Settings")
    
    # Search engine selection
    search_engines = {
        "Google": "google",
        "Bing": "bing",
        "Yahoo": "yahoo",
        "Yandex": "yandex",
        "DuckDuckGo": "duckduckgo",
        "Baidu": "baidu",
        "Ecosia": "ecosia",
        "YouTube": "youtube",
        "Wikipedia": "wikipedia",
        "Amazon": "amazon"
    }
    
    engine_choice = st.selectbox(
        "Search Engine",
        options=list(search_engines.keys()),
        index=0
    )
    
    # Analysis options
    st.markdown("### Analysis Options")
    
    analyze_people_also_ask = st.checkbox("People Also Ask", value=True)
    analyze_related_searches = st.checkbox("Related Searches", value=True)
    analyze_organic_results = st.checkbox("Organic Results", value=True)
    analyze_word_frequency = st.checkbox("Word Frequency", value=True)
    
    # Additional parameters
    st.markdown("### Additional Parameters")
    
    country = st.selectbox(
        "Country",
        options=["us", "uk", "ca", "au", "de", "fr", "es", "it", "jp", "br", "in"],
        index=0,
        help="Target country for search results"
    )
    
    language = st.selectbox(
        "Language",
        options=["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh"],
        index=0,
        help="Target language for search results"
    )
    
    num_results = st.slider(
        "Number of results to analyze",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Maximum number of results to include in analysis"
    )
    
    min_word_length = st.slider(
        "Minimum word length",
        min_value=2,
        max_value=10,
        value=3,
        help="Minimum length of words to include in analysis"
    )
    
    exclude_common_words = st.checkbox(
        "Exclude common words",
        value=True,
        help="Exclude common words like 'the', 'and', 'is', etc."
    )

    custom_stopwords = st.text_area(
        "Custom words to exclude (one per line)",
        value="",
        help="Add custom words to exclude from analysis"
    )

# Function to clean and process text
def process_text(text, min_length=3, exclude_common=True, custom_exclusions=None):
    if not text:
        return []
        
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    
    try:
        # Try to use NLTK tokenizer
        words = word_tokenize(text)
    except Exception:
        # Fallback to simple tokenization if NLTK fails
        words = text.split()
    
    # Get stopwords if needed
    stop_words = set()
    if exclude_common:
        try:
            # Try to get NLTK stopwords
            stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback to basic English stopwords
            basic_stopwords = {
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                'very', 'can', 'will', 'just', 'should', 'now'
            }
            stop_words = basic_stopwords
    
    # Add custom exclusions
    if custom_exclusions:
        stop_words.update(custom_exclusions)
    
    # Filter words
    filtered_words = [
        word for word in words 
        if len(word) >= min_length and word not in stop_words and word not in string.punctuation
    ]
    
    return filtered_words

# Function to get data from SerpAPI with error handling
def fetch_seo_data(query, engine_code, api_key, country="us", language="en", num_results=20):
    url = "https://serpapi.com/search"
    
    params = {
        "q": query,
        "engine": engine_code,
        "api_key": api_key,
        "gl": country,
        "hl": language,
        "num": num_results
    }
    
    try:
        with st.spinner(f"Fetching data from {engine_code.capitalize()}..."):
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
                return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Failed to parse response: {str(e)}")
        return None

# Function to create interactive word frequency charts with plotly
def create_word_freq_charts(word_df):
    # Bar chart with plotly
    fig_bar = px.bar(
        word_df.head(15),
        x='Count',
        y='Word',
        orientation='h',
        title="Top 15 Keywords by Frequency",
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig_bar.update_layout(
        height=500,
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        yaxis={'categoryorder':'total ascending'}
    )
    
    # Pie chart with plotly
    fig_pie = px.pie(
        word_df.head(10),
        values='Count',
        names='Word',
        title="Top 10 Keywords Distribution"
    )
    
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    fig_pie.update_layout(height=500)
    
    return fig_bar, fig_pie

# Function to create word cloud
def create_word_cloud(word_dict):
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=3
        ).generate_from_frequencies(word_dict)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        return fig
    except Exception as e:
        st.warning(f"Unable to generate word cloud: {str(e)}")
        # Return a placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Word Cloud Generation Failed", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, color='gray')
        ax.axis('off')
        return fig

# Function to analyze keyword competition
def analyze_competition(data):
    competition_metrics = {
        "total_results": data.get("search_information", {}).get("total_results", 0),
        "paid_results": len(data.get("ads", [])),
        "organic_results": len(data.get("organic_results", [])),
        "related_questions": len(data.get("related_questions", [])),
        "related_searches": len(data.get("related_searches", []))
    }
    
    # Calculate competition score (simplified)
    paid_weight = 0.4
    organic_weight = 0.3
    questions_weight = 0.15
    related_weight = 0.15
    
    max_paid = 10
    max_organic = 100
    max_questions = 10
    max_related = 10
    
    competition_score = (
        min(competition_metrics["paid_results"], max_paid) / max_paid * paid_weight +
        min(competition_metrics["organic_results"], max_organic) / max_organic * organic_weight +
        min(competition_metrics["related_questions"], max_questions) / max_questions * questions_weight +
        min(competition_metrics["related_searches"], max_related) / max_related * related_weight
    )
    
    return competition_metrics, min(competition_score * 100, 100)

# Main application UI
st.markdown('<h1 class="main-header">🔍 Advanced SEO Keyword Analyzer</h1>', unsafe_allow_html=True)

# Input area
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_area(
        "Enter a keyword or phrase to analyze",
        placeholder="e.g. best drones for photography",
        height=100
    )
with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

# Process when analyze button is clicked
if analyze_button and query.strip():
    # Get selected engine code
    engine_code = search_engines[engine_choice]
    
    # Custom stopwords processing
    custom_stop_list = []
    if custom_stopwords:
        custom_stop_list = [word.strip().lower() for word in custom_stopwords.split('\n') if word.strip()]
    
    # Fetch data
    data = fetch_seo_data(
        query=query,
        engine_code=engine_code,
        api_key=api_key,
        country=country,
        language=language,
        num_results=num_results
    )
    
    if data:
        # Display keyword competition metrics
        competition_metrics, competition_score = analyze_competition(data)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"## 📊 Keyword Competition Analysis for: '{query}'")
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Competition Score", f"{competition_score:.1f}%")
        with col2:
            st.metric("Total Results", f"{int(competition_metrics['total_results']):,}")
        with col3:
            st.metric("Paid Results", competition_metrics["paid_results"])
        with col4:
            st.metric("Organic Results", competition_metrics["organic_results"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Overview", 
            "❓ People Also Ask", 
            "🔗 Related Searches", 
            "🌐 Organic Results",
            "📝 Keyword Analysis"
        ])
        
        # Tab 1: Overview
        with tab1:
            st.markdown('<h3 class="subheader">Keyword Analysis Overview</h3>', unsafe_allow_html=True)
            
            # Extract and combine all text for analysis
            all_text = ""
            
            # Add titles and snippets from organic results
            if "organic_results" in data:
                for result in data.get("organic_results", []):
                    if "title" in result:
                        all_text += " " + result["title"]
                    if "snippet" in result:
                        all_text += " " + result["snippet"]
            
            # Add questions from People Also Ask
            if "related_questions" in data:
                for question in data.get("related_questions", []):
                    if "question" in question:
                        all_text += " " + question["question"]
            
            # Add related searches
            if "related_searches" in data:
                for search in data.get("related_searches", []):
                    if "query" in search:
                        all_text += " " + search["query"]
            
            # Process text and compute word frequencies
            words = process_text(
                all_text, 
                min_length=min_word_length,
                exclude_common=exclude_common_words,
                custom_exclusions=custom_stop_list
            )
            
            word_counts = Counter(words)
            
            if word_counts:
                # Create word cloud
                st.markdown("### 📊 Word Cloud")
                word_cloud_fig = create_word_cloud(word_counts)
                st.pyplot(word_cloud_fig)
                
                # Create summary metrics
                st.markdown("### 📈 Key Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Keywords", len(word_counts))
                with col2:
                    st.metric("Unique Words", len(set(words)))
                with col3:
                    most_common = word_counts.most_common(1)[0][0] if word_counts else "-"
                    st.metric("Most Common Keyword", most_common)
            else:
                st.warning("Not enough text data to analyze word frequencies")
        
        # Tab 2: People Also Ask
        with tab2:
            st.markdown('<h3 class="subheader">People Also Ask</h3>', unsafe_allow_html=True)
            
            if "related_questions" in data and data["related_questions"]:
                questions = []
                for i, item in enumerate(data["related_questions"]):
                    if "question" in item:
                        questions.append({
                            "Question": item["question"],
                            "Position": i + 1
                        })
                
                if questions:
                    # Display as dataframe
                    questions_df = pd.DataFrame(questions)
                    st.dataframe(questions_df, use_container_width=True)
                    
                    # Visualize question length distribution
                    question_lengths = [len(q["Question"]) for q in questions]
                    fig = px.histogram(
                        x=question_lengths,
                        nbins=20,
                        title="Question Length Distribution",
                        labels={"x": "Character Count", "y": "Frequency"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    st.download_button(
                        "Download Questions CSV",
                        questions_df.to_csv(index=False).encode('utf-8'),
                        f"{query}_questions.csv",
                        "text/csv",
                        key='download-questions'
                    )
                else:
                    st.info("No questions found in the 'People Also Ask' section.")
            else:
                st.info("No 'People Also Ask' data available for this query.")
        
        # Tab 3: Related Searches
        with tab3:
            st.markdown('<h3 class="subheader">Related Searches</h3>', unsafe_allow_html=True)
            
            if "related_searches" in data and data["related_searches"]:
                related = []
                for i, item in enumerate(data["related_searches"]):
                    if "query" in item:
                        related.append({
                            "Related Search": item["query"],
                            "Position": i + 1
                        })
                
                if related:
                    # Display as dataframe
                    related_df = pd.DataFrame(related)
                    st.dataframe(related_df, use_container_width=True)
                    
                    # Download option
                    st.download_button(
                        "Download Related Searches CSV",
                        related_df.to_csv(index=False).encode('utf-8'),
                        f"{query}_related_searches.csv",
                        "text/csv",
                        key='download-related'
                    )
                else:
                    st.info("No related searches found.")
            else:
                st.info("No related searches data available for this query.")
        
        # Tab 4: Organic Results
        with tab4:
            st.markdown('<h3 class="subheader">Top Organic Search Results</h3>', unsafe_allow_html=True)
            
            if "organic_results" in data and data["organic_results"]:
                organic_results = []
                for i, res in enumerate(data["organic_results"]):
                    result = {
                        "Position": i + 1,
                        "Title": res.get("title", "N/A"),
                        "URL": res.get("link", "N/A"),
                        "Snippet": res.get("snippet", "N/A")[:100] + "..." if res.get("snippet") else "N/A",
                    }
                    organic_results.append(result)
                
                if organic_results:
                    # Display as dataframe
                    organic_df = pd.DataFrame(organic_results)
                    st.dataframe(organic_df, use_container_width=True, column_config={
                        "URL": st.column_config.LinkColumn(),
                        "Title": st.column_config.TextColumn(width="large"),
                        "Snippet": st.column_config.TextColumn(width="large")
                    })
                    
                    # Domain analysis
                    if organic_results:
                        domains = []
                        for result in organic_results:
                            url = result["URL"]
                            if url != "N/A":
                                domain = url.split('//')[-1].split('/')[0]
                                domains.append(domain)
                        
                        domain_counts = Counter(domains)
                        domain_df = pd.DataFrame({
                            "Domain": list(domain_counts.keys()),
                            "Count": list(domain_counts.values())
                        }).sort_values(by="Count", ascending=False)
                        
                        st.markdown("### 🌐 Domain Distribution")
                        fig = px.pie(
                            domain_df,
                            values='Count',
                            names='Domain',
                            title="Domain Distribution in Search Results"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download option
                    st.download_button(
                        "Download Organic Results CSV",
                        organic_df.to_csv(index=False).encode('utf-8'),
                        f"{query}_organic_results.csv",
                        "text/csv",
                        key='download-organic'
                    )
                else:
                    st.info("No organic results found.")
            else:
                st.info("No organic results data available for this query.")
        
        # Tab 5: Keyword Analysis
        with tab5:
            st.markdown('<h3 class="subheader">Keyword Analysis</h3>', unsafe_allow_html=True)
            
            if word_counts:
                word_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)
                
                # Show interactive table
                st.dataframe(word_df, use_container_width=True)
                
                # Visualizations
                fig_bar, fig_pie = create_word_freq_charts(word_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Download option
                st.download_button(
                    "Download Keyword Analysis CSV",
                    word_df.to_csv(index=False).encode('utf-8'),
                    f"{query}_keyword_analysis.csv",
                    "text/csv",
                    key='download-keywords'
                )
            else:
                st.info("Not enough data to perform keyword analysis.")
        
    else:
        st.error("Failed to retrieve data. Please check your API key and try again.")
else:
    if analyze_button and not query.strip():
        st.warning("Please enter a keyword or phrase to analyze.")
    else:
        # Show intro content when app loads
        st.markdown("""
        <div class="card">
            <h3>Welcome to the Advanced SEO Keyword Analyzer! 👋</h3>
            <p>This tool helps you analyze keywords and get insights for your SEO strategy. Enter a keyword or phrase in the text area above and click "Analyze" to get started.</p>
            <p>You can customize the analysis settings in the sidebar on the left.</p>
            <h4>Features:</h4>
            <ul>
                <li>Analyze keywords across multiple search engines</li>
                <li>Discover related questions and searches</li>
                <li>View top organic search results</li>
                <li>Analyze word frequency and competition</li>
                <li>Generate visualizations and reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "Data updated in real-time"
)

# Requirements for this app:
# pip install streamlit requests pandas matplotlib seaborn nltk wordcloud plotly
#
# Note: If you experience issues with NLTK or WordCloud, you can comment out those
# dependencies and the code will fall back to simpler implementations.